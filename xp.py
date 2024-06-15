from collections import defaultdict
from typing import Dict, List, Optional
import traceback
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from rich import print
import rich.progress as progress
from rich.console import Console
from transformers import BertTokenizerFast
from renard.pipeline.core import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from tibert.bertcoref import (
    BertForCoreferenceResolution,
    CoreferenceDataset,
    CoreferenceDocument,
)
from dataset_ingredients import litbank_ingredient, load_litbank
from splice.data import Novel
from splice.metrics import (
    align_characters,
    score_ner,
    score_network_extraction_edges,
    score_character_unification,
    score_coref,
)
from splice.corefs import train_coref_model
from splice.sacred_utils import (
    archive_huggingface_model,
    archive_pipeline_state_,
    archive_graph_,
)
from splice.graph_utils import shared_layout
from splice.utils import mean_noNaN

ex = Experiment("full_pipeline", ingredients=[litbank_ingredient])
ex.observers.append(FileStorageObserver("./runs"))


@ex.config
def config():
    # minimum number of nodes in a gold graph needed to consider graph
    # measures. Novels that do *not* satisfy this number of nodes are
    # used to train the coreference model.
    min_graph_nodes: int = 10

    # coreference model to use. When given the empty string, will
    # train a model from scratch.
    coref_model_id: str = ""

    # max distance between two mention for them to be in
    # co-occurrence, in tokens
    co_occurrences_dist: int = 32

    # whether to use hierarchical merging
    hierarchical_merging: bool = False


@ex.automain
def main(
    _run: Run,
    min_graph_nodes: int,
    coref_model_id: str,
    co_occurrences_dist: int,
    hierarchical_merging: bool,
):
    print_config(_run)

    # NOTE: avoid issues with tkinter not finding main thread
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")

    novels: List[Novel] = load_litbank()  # type: ignore
    _run.info["novels"] = [novel.title for novel in novels]

    co_occ_kwargs = {"co_occurrences_dist": (co_occurrences_dist, "tokens")}

    gold_pipeline = Pipeline(
        [CoOccurrencesGraphExtractor(**co_occ_kwargs)],
        progress_report=None,
        warn=False,
    )

    progress_console = Console()

    # First, compute all gold networks
    outputs_gold = {}
    for novel in progress.track(novels, console=progress_console):
        progress_console.print(f"extracting {novel.title} gold network...", end="")

        try:
            out_gold = gold_pipeline(
                tokens=novel.tokens,
                sentences=novel.sents,
                entities=novel.entities,
                corefs=novel.corefs,
                characters=novel.characters,
            )
            archive_pipeline_state_(_run, out_gold, f"{novel.title}_state_gold")
        except Exception as e:
            progress_console.print(f"error {e}")
            continue

        outputs_gold[novel.title] = out_gold

        progress_console.print("done!")

    title2novel = {novel.title: novel for novel in novels}
    analysis_selected_novels = [
        title2novel[title]
        for title, out in outputs_gold.items()
        if len(out.character_network.nodes) >= min_graph_nodes
    ]
    coref_train_novels = [
        title2novel[title]
        for title in set(outputs_gold.keys())
        - set([novel.title for novel in analysis_selected_novels])
    ]

    _run.info["analysis_novels"] = [novel.title for novel in analysis_selected_novels]

    # train Renard coref model if necessary
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    if coref_model_id == "":
        max_span_size = 10
        coref_train_dataset = CoreferenceDataset(
            [
                CoreferenceDocument(novel.tokens, novel.corefs)
                for novel in coref_train_novels
            ],
            tokenizer,
            max_span_size,
        )
        coref_train_dataset.limit_doc_size_tokens_(750)
        coref_model = train_coref_model(_run, tokenizer, coref_train_dataset)
        archive_huggingface_model(_run, coref_model, "coref_model")
    else:
        coref_model = BertForCoreferenceResolution.from_pretrained(coref_model_id)

    # instantiate prediction pipelines
    if hierarchical_merging:
        pipeline = Pipeline(
            [
                BertNamedEntityRecognizer(),
                BertCoreferenceResolver(
                    model=coref_model,
                    tokenizer=tokenizer,
                    block_size=512,
                    hierarchical_merging=True,
                ),
                GraphRulesCharacterUnifier(link_corefs_mentions=True),
                CoOccurrencesGraphExtractor(**co_occ_kwargs),
            ],
            progress_report=None,
            warn=False,
        )
    else:
        pipeline = Pipeline(
            [
                BertNamedEntityRecognizer(),
                BertCoreferenceResolver(
                    model=coref_model,
                    tokenizer=tokenizer,
                    block_size=99999,  # HACK
                ),
                GraphRulesCharacterUnifier(link_corefs_mentions=True),
                CoOccurrencesGraphExtractor(**co_occ_kwargs),
            ],
            progress_report=None,
            warn=False,
        )

    pipeline_nocoref = Pipeline(
        [
            BertNamedEntityRecognizer(),
            GraphRulesCharacterUnifier(),
            CoOccurrencesGraphExtractor(**co_occ_kwargs),
        ],
        progress_report=None,
        warn=False,
    )

    all_metrics = defaultdict(list)

    def store_log_(
        novel: Novel, metric: str, value: float, pipeline_name: Optional[str] = None
    ) -> Dict[str, list]:
        """Store a metric value in all_metrics, and log it in the current sacred run"""
        if not pipeline_name is None:
            metric_key = f"{pipeline_name}.{metric}"
        else:
            metric_key = metric
        _run.log_scalar(f"{novel.title}.{metric_key}", value)
        all_metrics[metric_key].append(value)
        return all_metrics

    # use prediction pipelines and compute task+graph metrics
    for novel in progress.track(analysis_selected_novels, console=progress_console):
        progress_console.print(f"processing {novel.title}...", end="")

        try:
            out = pipeline(tokens=novel.tokens, sentences=novel.sents)
            archive_pipeline_state_(_run, out, f"{novel.title}_state")
        except Exception as e:
            progress_console.print(f"error {e}")
            print(traceback.format_exc())
            continue

        try:
            out_nocoref = pipeline_nocoref(tokens=novel.tokens, sentences=novel.sents)
            archive_pipeline_state_(_run, out_nocoref, f"{novel.title}_state_nocoref")
        except Exception as e:
            progress_console.print(f"error {e}")
            print(traceback.format_exc())
            continue

        out_gold = outputs_gold[novel.title]

        # NER metrics
        # -----------
        ner_precision, ner_recall, ner_f1 = score_ner(
            novel.tokens,
            out.entities,
            out_gold.entities,
            ignore_classes={"LOC", "ORG", "MISC"},
        )
        store_log_(novel, "ner_precision", ner_precision)
        store_log_(novel, "ner_recall", ner_recall)
        store_log_(novel, "ner_f1", ner_f1)

        # Coref metrics
        # -------------
        coref_metrics = score_coref(novel.tokens, out.corefs, out_gold.corefs)
        for metric_name, metric_dict in coref_metrics.items():
            for metric_key, value in metric_dict.items():
                store_log_(novel, f"{metric_name}_{metric_key}", value)

        # Character unification / graph metrics
        # -------------------------------------
        for pipeline_name, out_pipeline in [
            ("pipeline", out),
            ("pipeline_nocoref", out_nocoref),
        ]:
            node_precision, node_recall, node_f1 = score_character_unification(
                [character.names for character in out_gold.characters],
                [character.names for character in out_pipeline.characters],
            )
            store_log_(novel, "node_precision", node_precision, pipeline_name)
            store_log_(novel, "node_recall", node_recall, pipeline_name)
            store_log_(novel, "node_f1", node_f1, pipeline_name)

            mapping, _ = align_characters(out_gold.characters, out_pipeline.characters)
            edge_precision, edge_recall, edge_f1 = score_network_extraction_edges(
                out_gold.character_network, out_pipeline.character_network, mapping
            )
            store_log_(novel, "edge_precision", edge_precision, pipeline_name)
            store_log_(novel, "edge_recall", edge_recall, pipeline_name)
            store_log_(novel, "edge_f1", edge_f1, pipeline_name)

            w_edge_precision, w_edge_recall, w_edge_f1 = score_network_extraction_edges(
                out_gold.character_network,
                out_pipeline.character_network,
                mapping,
                weighted=True,
            )
            store_log_(
                novel, "weighted_edge_precision", w_edge_precision, pipeline_name
            )
            store_log_(novel, "weighted_edge_recall", w_edge_recall, pipeline_name)
            store_log_(novel, "weighted_edge_f1", w_edge_f1, pipeline_name)

        mappings = [
            align_characters(out_gold.characters, out.characters)[1]
            for out in [out, out_nocoref]
        ]

        layout = shared_layout(
            out_gold.character_network,
            [out.character_network, out_nocoref.character_network],
            mappings,
        )
        archive_graph_(_run, out_gold, f"{novel.title}.network_gold", layout)
        archive_graph_(_run, out, f"{novel.title}.pipeline.network", layout)
        archive_graph_(
            _run, out_nocoref, f"{novel.title}.pipeline_nocoref.network", layout
        )

        progress_console.print("done!")

    # store mean metrics
    for key, values in all_metrics.items():
        _run.log_scalar(f"MEAN_{key}", mean_noNaN(values))
