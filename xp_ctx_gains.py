import traceback
from typing import Dict, List, Literal, Optional
from collections import defaultdict
from sacred import Experiment
from sacred.run import Run
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from rich import print
import rich.progress as progress
from rich.console import Console
from renard.pipeline.core import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.ner.retrieval import (
    NEREnsembleContextRetriever,
    NERSamenounContextRetriever,
    NERBM25ContextRetriever,
    NERNeighborsContextRetriever,
)
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from torch import select_scatter
from dataset_ingredients import litbank_ingredient, load_litbank
from splice.data import Novel
from splice.sacred_utils import archive_pipeline_state_, archive_graph_
from splice.metrics import (
    align_characters,
    score_ner,
    score_network_extraction_edges,
    score_character_unification,
    score_coref,
)
from splice.graph_utils import shared_layout
from splice.utils import mean_noNaN


ex = Experiment("aug_gains", ingredients=[novelties_ingredient])
ex.observers.append(FileStorageObserver("./runs"))


@ex.config
def config():
    # minimum number of nodes in a gold graph needed to consider graph
    # measures. Novels that do *not* satisfy this number of nodes are
    # used to train the coreference model.
    min_graph_nodes: int = 10

    # max distance between two mention for them to be in
    # co-occurrence, in tokens
    co_occurrences_dist: int = 32

    # id of the coreference model
    coref_model_id: str = "compnet-renard/bert-base-cased-literary-coref"


@ex.automain
def main(
    _run: Run,
    min_graph_nodes: int,
    co_occurrences_dist: int,
    coref_model_id: str,
):
    print_config(_run)

    novels: List[Novel] = load_novelties()  # type: ignore
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

    # Select analysis novels by filtering networks with less than
    # min_graph_nodes nodes
    title2novel = {novel.title: novel for novel in novels}
    analysis_selected_novels = [
        title2novel[title]
        for title, out in outputs_gold.items()
        if len(out.character_network.nodes) >= min_graph_nodes
    ]

    pipelines = {
        "-ctx": lambda: Pipeline(
            [
                BertNamedEntityRecognizer(),
                GraphRulesCharacterUnifier(),
                CoOccurrencesGraphExtractor(**co_occ_kwargs),
            ],
            progress_report=None,
            warn=False,
        ),
        "+ctx": lambda: Pipeline(
            [
                BertNamedEntityRecognizer(
                    context_retriever=NEREnsembleContextRetriever(
                        [
                            # NERSamenounContextRetriever(8),
                            # NERBM25ContextRetriever(8),
                            # NERNeighborsContextRetriever(8 * 2),
                            NERSamenounContextRetriever(4),
                            NERBM25ContextRetriever(4),
                            NERNeighborsContextRetriever(4 * 2),
                        ],
                        k=1,
                    )
                ),
                GraphRulesCharacterUnifier(),
                CoOccurrencesGraphExtractor(**co_occ_kwargs),
            ],
            progress_report=None,
            warn=False,
        ),
    }

    for pipeline_name, make_pipeline in pipelines.items():

        all_metrics = defaultdict(list)

        def store_log_(
            novel: Novel, metric: str, value: float, pipeline_name: str
        ) -> Dict[str, list]:
            """Store a metric value in all_metrics, and log it in the current sacred run"""
            metric_key = f"{pipeline_name}.{metric}"
            _run.log_scalar(f"{novel.title}.{metric_key}", value)
            all_metrics[metric_key].append(value)
            return all_metrics

        pipeline = make_pipeline()

        # use prediction pipelines and compute task+graph metrics
        for novel in progress.track(analysis_selected_novels, console=progress_console):
            out_gold = outputs_gold[novel.title]

            progress_console.print(
                f"{pipeline_name}: processing {novel.title}...", end=""
            )

            try:
                out = pipeline(tokens=novel.tokens, sentences=novel.sents)
                archive_pipeline_state_(
                    _run, out, f"{pipeline_name}_{novel.title}_state"
                )
            except Exception as e:
                progress_console.print(f"{pipeline_name}: error {e}")
                print(traceback.format_exc())
                continue

            # NER metrics
            # -----------
            ner_precision, ner_recall, ner_f1 = score_ner(
                novel.tokens,
                out.entities,
                out_gold.entities,
                ignore_classes={"LOC", "ORG", "MISC"},
            )
            store_log_(novel, "ner_precision", ner_precision, pipeline_name)
            store_log_(novel, "ner_recall", ner_recall, pipeline_name)
            store_log_(novel, "ner_f1", ner_f1, pipeline_name)

            # Character unification / graph metrics
            # -------------------------------------
            node_precision, node_recall, node_f1 = score_character_unification(
                [character.names for character in out_gold.characters],
                [character.names for character in out.characters],
            )
            store_log_(novel, "node_precision", node_precision, pipeline_name)
            store_log_(novel, "node_recall", node_recall, pipeline_name)
            store_log_(novel, "node_f1", node_f1, pipeline_name)

            mapping, _ = align_characters(out_gold.characters, out.characters)
            edge_precision, edge_recall, edge_f1 = score_network_extraction_edges(
                out_gold.character_network, out.character_network, mapping
            )
            store_log_(novel, "edge_precision", edge_precision, pipeline_name)
            store_log_(novel, "edge_recall", edge_recall, pipeline_name)
            store_log_(novel, "edge_f1", edge_f1, pipeline_name)

            w_edge_precision, w_edge_recall, w_edge_f1 = score_network_extraction_edges(
                out_gold.character_network,
                out.character_network,
                mapping,
                weighted=True,
            )
            store_log_(
                novel, "weighted_edge_precision", w_edge_precision, pipeline_name
            )
            store_log_(novel, "weighted_edge_recall", w_edge_recall, pipeline_name)
            store_log_(novel, "weighted_edge_f1", w_edge_f1, pipeline_name)

            progress_console.print("done!")

        # store mean metrics
        for key, values in all_metrics.items():
            k = f"MEAN_{key}"
            m = mean_noNaN(values)
            print(f"{k}: {m}")
            _run.log_scalar(k, m)

    import pandas as pd

    ids = [novel.title for novel in analysis_selected_novels]
    df = pd.DataFrame(
        {
            "-coref-ctx.node_f1": [
                f"{novel.title}.-coref-ctx.node_f1" for novel in ids
            ],
            "-coref-ctx.ner_f1": [f"{novel.title}.-coref-ctx.ner_f1" for novel in ids],
            "-coref+ctx.node_f1": [
                f"{novel.title}.-coref+ctx.node_f1" for novel in ids
            ],
            "-coref+ctx.ner_f1": [f"{novel.title}.-coref+ctx.ner_f1" for novel in ids],
        }
    )
    df = df.set_index(ids)
    print(df)
