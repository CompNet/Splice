# -*- eval: (code-cells-mode); -*-

# %%
import json, pickle
import networkx as nx
import pandas as pd
from rich.console import Console
import rich.progress as progress
from dataset_ingredients import load_litbank
from splice.data import Novel
from splice.metrics import (
    score_character_unification,
    score_network_extraction_edges,
    align_characters,
)
from renard.pipeline.core import Pipeline
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

from splice.utils import mean_noNaN


RUN = "main_xp_32"

with open(f"./runs/{RUN}/info.json") as f:
    info = json.load(f)
analysis_novels = set(info["analysis_novels"])

novels_coref: list[Novel] = load_litbank(
    "/home/aethor/Dev/litbank", "./flat_litbank_ner", keep_only_NER_mentions=False
)
novels_coref = [n for n in novels_coref if n.title in analysis_novels]

novels_nocoref: list[Novel] = load_litbank(
    "/home/aethor/Dev/litbank", "./flat_litbank_ner", keep_only_NER_mentions=True
)
novels_nocoref = [n for n in novels_nocoref if n.title in analysis_novels]

pipeline = Pipeline(
    [CoOccurrencesGraphExtractor(co_occurrences_dist=32)],
    progress_report=None,
    warn=False,
)

progress_console = Console()

edge_recalls = []
w_edge_precisions = []
w_edge_recalls = []

for novel_coref, novel_nocoref in progress.track(
    zip(novels_coref, novels_nocoref), console=progress_console
):
    out_nocoref = pipeline(
        tokens=novel_nocoref.tokens,
        sentences=novel_nocoref.sents,
        entities=novel_nocoref.entities,
        characters=novel_nocoref.characters,
    )

    # hum?
    out_coref = pipeline(
        tokens=novel_coref.tokens,
        sentences=novel_coref.sents,
        entities=novel_coref.entities,
        corefs=novel_coref.corefs,
        characters=novel_coref.characters,
    )

    node_precision, node_recall, node_f1 = score_character_unification(
        [character.names for character in out_coref.characters],
        [character.names for character in out_nocoref.characters],
    )
    progress_console.print(f"{node_precision=:.2f}, {node_recall=:.2f}, {node_f1=:.2f}")

    mapping, _ = align_characters(out_coref.characters, out_nocoref.characters)
    edge_precision, edge_recall, edge_f1 = score_network_extraction_edges(
        out_coref.character_network, out_nocoref.character_network, mapping
    )
    progress_console.print(f"{edge_precision=:.2f}, {edge_recall=:.2f}, {edge_f1=:.2f}")
    edge_recalls.append(edge_recall)

    w_edge_precision, w_edge_recall, w_edge_f1 = score_network_extraction_edges(
        out_coref.character_network,
        out_nocoref.character_network,
        mapping,
        weighted=True,
    )
    progress_console.print(
        f"{w_edge_precision=:.2f}, {w_edge_recall=:.2f}, {w_edge_f1=:.2f}"
    )
    w_edge_precisions.append(w_edge_precision)
    w_edge_recalls.append(w_edge_recall)


mean_edge_recall = mean_noNaN(edge_recalls)
mean_w_edge_recall = mean_noNaN(w_edge_recalls)
mean_w_edge_precision = mean_noNaN(w_edge_precisions)
print(f"mean edge recall: {round(mean_edge_recall * 100, 2)}")
print(f"mean weighted edge precision: {round(mean_w_edge_precision * 100, 2)}")
print(f"mean weighted edge recall: {round(mean_w_edge_recall * 100, 2)}")
