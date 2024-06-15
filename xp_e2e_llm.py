from typing import Any, List, Dict, Literal, Optional, Set
import json, re, pickle
from collections import defaultdict
import pathlib as pl
import transformers
from huggingface_hub import login
import torch
from openai import OpenAI
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from rich import print as printr
import rich.progress as progress
from rich.console import Console
from dataset_ingredients import litbank_ingredient
from renard.pipeline.core import PipelineStep, Pipeline, Mention
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from renard.pipeline.character_unification import Character
from dataset_ingredients import load_litbank
from splice.data import Novel
from splice.metrics import (
    score_character_unification,
    align_characters,
    score_network_extraction_edges,
)
from splice.sacred_utils import archive_text
from splice.utils import mean_noNaN


ex = Experiment("llm_pipeline", ingredients=[litbank_ingredient])
ex.observers.append(FileStorageObserver("./runs"))


@ex.config
def config():
    # main xp directory
    input_dir: str

    # one of: "gpt3.5", "gpt4o", "llama3-8b-instruct"
    model: str

    # if model == "gpt3.5"
    openAI_API_key: str = ""

    # if model == "llama3-8b-instruct"
    hg_access_token: str = ""

    # one of "cuda", "cpu", "auto"
    device: str = "cuda"


class LLMCharacterUnifier(PipelineStep):
    SYSTEM_PROMPT = "You are an expert in literature and natural langage processing."

    USER_PROMPT = r"""Given a text, you must extract characters and their mentions. Your answer must be the original text, where character mentions are tagged with the following format: [CHARACTER_ID]CHARACTER MENTION[/CHARACTER_ID]. You must tag characters mentions only.

Here are some examples of this task:

Example 1:

Input:
Elric was riding his horse . Alongside Moonglum , the prince of ruins was looking for his dark sword .

Output:
[0] Elric [/0] was riding [0] his [/0] horse . Alongside [1] Moonglum [/1] , the [0] prince of ruins [/0] was looking for [0] his [/0] dark sword .


Example 2:

Input:
Princess Liana felt sad , because Zarth Arn was gone . The princess decided she should sleep .

Output:
[0] Princess Liana [/0] felt sad , because [1] Zarth Arn [/1] was gone . [0] The princess [/0] decided [0] she [/0] should sleep .
        """

    OPENAI_NAME2MODEL = {"gpt3.5": "gpt-3.5-turbo-0125", "gpt4o": "gpt-4o-2024-05-13"}

    def __init__(
        self,
        model: Literal["gpt3.5", "gpt4o", "llama3-8b-instruct"],
        device: Literal["auto", "cuda", "cpu"] = "auto",
        openAI_API_key: Optional[str] = None,
        hg_access_token: Optional[str] = None,
    ) -> None:
        self.model = model

        self.device = device

        if self.model in ["gpt3.5", "gpt4o"]:
            assert not openAI_API_key is None
        self.openAI_API_key = openAI_API_key

        if self.model == "llama3-8b-instruct":
            assert not hg_access_token is None
        self.hg_access_token = hg_access_token

        self.pipeline = None

        super().__init__()

    def _pipeline_init_(self, lang: str, progress_reporter):
        if self.model == "llama3-8b-instruct":
            if self.pipeline is None:
                login(self.hg_access_token)
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map=self.device,
                )
        super()._pipeline_init_(lang, progress_reporter)

    def _call_openai(self, tokens: List[str]) -> Optional[str]:
        openai_client = OpenAI(api_key=self.openAI_API_key)

        answer = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": LLMCharacterUnifier.SYSTEM_PROMPT},
                {"role": "user", "content": LLMCharacterUnifier.USER_PROMPT},
                {
                    "role": "user",
                    "content": "Input:\n{}".format(" ".join(tokens))
                    + "\n\n\nOutput:\n",
                },
            ],
            model=LLMCharacterUnifier.OPENAI_NAME2MODEL[self.model],
            max_tokens=4096,
        )

        return answer.choices[0].message.content

    def _call_llama3(self, tokens: List[str]) -> str:
        assert not self.pipeline is None

        messages = [
            {"role": "system", "content": LLMCharacterUnifier.SYSTEM_PROMPT},
            {"role": "user", "content": LLMCharacterUnifier.USER_PROMPT},
            {
                "role": "user",
                "content": "Input:\n{}".format(" ".join(tokens)) + "\n\n\nOutput:\n",
            },
        ]
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=4096,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][-1]["content"]

    @staticmethod
    def is_surely_a_name(candidate_name_tokens: List[str]) -> bool:
        def is_pronoun(token: str):
            return token.lower() in {
                "his",
                "her",
                "he",
                "she",
                "it",
                "they",
                "them",
                "their",
                "you",
            }

        return all(t[0].isupper() for t in candidate_name_tokens) and not all(
            is_pronoun(t) for t in candidate_name_tokens
        )

    @staticmethod
    def parse_llm_output(output_tokens: List[str]) -> List[Character]:
        characters = {}
        in_tag = False
        interpreted_output_tokens = []
        in_tag_tokens = []
        for token in output_tokens:
            m = re.match(r"\[\/?([0-9]+)\]", token)
            if in_tag:
                if m:
                    char_id = int(m.group(1))
                    character = characters.get(char_id, Character(frozenset(), []))
                    start = len(interpreted_output_tokens) - len(in_tag_tokens)
                    end = len(interpreted_output_tokens)
                    names = character.names
                    if LLMCharacterUnifier.is_surely_a_name(in_tag_tokens):
                        alias = " ".join(in_tag_tokens)
                        names = names.union({alias})
                    character = Character(
                        names,
                        character.mentions + [Mention(in_tag_tokens, start, end)],
                    )
                    characters[char_id] = character
                    in_tag_tokens = []
                    in_tag = False
                else:
                    in_tag_tokens.append(token)
                    interpreted_output_tokens.append(token)
            else:
                if m:
                    in_tag = True
                else:
                    interpreted_output_tokens.append(token)

        # we filter characters that have ne found names, since these
        # are often noise
        return [c for c in characters.values() if len(c.names) > 0]

    def __call__(self, tokens: List[str], **kwargs) -> Dict[str, Any]:
        if self.model == "llama3-8b-instruct":
            output_raw = self._call_llama3(tokens)
        elif self.model in ["gpt3.5", "gpt4o"]:
            output_raw = self._call_openai(tokens)
        else:
            raise ValueError(f"unkown model: {self.model}")

        if output_raw is None:
            return {"_llm_annotated_text": None, "characters": []}

        output_tokens = output_raw.split(" ")
        return {
            "_llm_annotated_text": output_raw,
            "characters": LLMCharacterUnifier.parse_llm_output(output_tokens),
            # NOTE: we actively modify the 'tokens' state of the
            # pipeline. The LLM might hallucinates, so this ensures
            # that 'characters' are in line with 'tokens' for the
            # graph extractor.
            "tokens": output_tokens,
        }

    def production(self) -> Set[str]:
        return {"characters"}

    def needs(self) -> Set[str]:
        return {"tokens"}


@ex.automain
def main(
    _run: Run,
    input_dir: str,
    model: Literal["gpt3.5", "llama3-8b-instruct"],
    openAI_API_key: str,
    hg_access_token: str,
    device: Literal["auto", "cuda", "cpu"],
):
    print_config(_run)

    RUN_PATH = pl.Path(input_dir)

    with open(RUN_PATH / "info.json") as f:
        info = json.load(f)

    with open(RUN_PATH / "run.json") as f:
        run_dict = json.load(f)
    co_occurrences_dist = run_dict["meta"]["config_updates"]["co_occurrences_dist"]

    novels: List[Novel] = load_litbank()  # type: ignore
    novels = [n for n in novels if n.title in info["analysis_novels"]]
    _run.info["novels"] = [novel.title for novel in novels]

    pipeline = Pipeline(
        [
            LLMCharacterUnifier(
                model,
                device,
                openAI_API_key=openAI_API_key,
                hg_access_token=hg_access_token,
            ),
            CoOccurrencesGraphExtractor(
                co_occurrences_dist=(co_occurrences_dist, "tokens")
            ),
        ],
        progress_report=None,
        warn=False,
    )

    all_metrics = defaultdict(list)

    def store_log_(novel: Novel, metric: str, value: float) -> Dict[str, list]:
        """Store a metric value in all_metrics, and log it in the current sacred run"""
        _run.log_scalar(f"{novel.title}.{metric}", value)
        all_metrics[metric].append(value)
        return all_metrics

    progress_console = Console()
    for novel in progress.track(novels):
        progress_console.print(
            f"extracting [green]{novel.title}[/green] network...", end=""
        )

        out_pipeline = pipeline(tokens=novel.tokens, sentences=[novel.tokens])

        archive_text(
            _run, out_pipeline._llm_annotated_text, f"{novel.title}_llm_annotated_text"
        )

        with open(RUN_PATH / f"{novel.title}_state_gold.pickle", "rb") as f:
            out_gold = pickle.load(f)

        node_precision, node_recall, node_f1 = score_character_unification(
            [character.names for character in out_gold.characters],
            [character.names for character in out_pipeline.characters],
        )
        store_log_(novel, "node_precision", node_precision)
        store_log_(novel, "node_recall", node_recall)
        store_log_(novel, "node_f1", node_f1)

        mapping, _ = align_characters(out_gold.characters, out_pipeline.characters)
        edge_precision, edge_recall, edge_f1 = score_network_extraction_edges(
            out_gold.character_network, out_pipeline.character_network, mapping
        )
        store_log_(novel, "edge_precision", edge_precision)
        store_log_(novel, "edge_recall", edge_recall)
        store_log_(novel, "edge_f1", edge_f1)

        w_edge_precision, w_edge_recall, w_edge_f1 = score_network_extraction_edges(
            out_gold.character_network,
            out_pipeline.character_network,
            mapping,
            weighted=True,
        )
        store_log_(novel, "weighted_edge_precision", w_edge_precision)
        store_log_(novel, "weighted_edge_recall", w_edge_recall)
        store_log_(novel, "weighted_edge_f1", w_edge_f1)

        progress_console.print(f"done!")

    # store mean metrics
    for key, values in all_metrics.items():
        _run.log_scalar(f"MEAN_{key}", mean_noNaN(values))
