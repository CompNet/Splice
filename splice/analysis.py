from __future__ import annotations
from typing import Callable, Literal, Optional, Tuple, List
import math
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from renard.pipeline.core import Pipeline, PipelineState, PipelineStep
from renard.pipeline.ner import NEREntity
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from tibert.bertcoref import CoreferenceDocument
from tibert.score import (
    score_muc,
    score_b_cubed,
    score_ceaf,
    score_blanc,
    score_lea,
    score_mention_detection,
)
from splice.data import extract_characters
from splice.corefs import deteriorate_coref
from splice.metrics import (
    score_character_unification,
    align_characters,
    score_network_extraction_edges,
)
from splice.ner import entities_to_BIO, deteriorate_ner


class SpliceNaiveCharacterUnifier(PipelineStep):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, entities, corefs, **kwargs) -> dict:
        return {"characters": extract_characters(entities, corefs)}

    def needs(self) -> set:
        return {"entities", "corefs"}

    def production(self) -> set:
        return {"characters"}


def compute_metrics_over_coref_degradation(
    state_gold: PipelineState,
    character_unification_step: Literal["naive", "rules"],
    character_unification_step_kwargs: dict,
    report_frequency: float,
    degradation_steps: int,
    co_occurrences_dist: Tuple[int, Literal["tokens", "sentences"]],
    degradation_actions: Optional[list[Callable]] = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    :return: coref_metrics_dict, graph_metrics_dict
    """
    cu_step = (
        GraphRulesCharacterUnifier(**character_unification_step_kwargs)
        if character_unification_step == "rules"
        else SpliceNaiveCharacterUnifier(**character_unification_step_kwargs)
    )

    # as in xp.py
    # NOTE: SpliceNaiveCharacterUnifier reproduces the process of
    # extracting a character from entities+corefs as done when loading
    # data.
    pipeline = Pipeline(
        [
            cu_step,
            CoOccurrencesGraphExtractor(co_occurrences_dist=co_occurrences_dist),
        ],
        progress_report=None,
    )
    # necessary since we only use pipeline.rerun_from
    pipeline._pipeline_init_steps()

    state = deepcopy(state_gold)
    assert not state.tokens is None
    assert not state_gold.corefs is None

    corefs_pred = CoreferenceDocument(state.tokens, state_gold.corefs)  # type: ignore
    corefs_refs = CoreferenceDocument(state.tokens, state_gold.corefs)  # type: ignore

    coref_metrics_dict = defaultdict(list)
    graph_metrics_dict = defaultdict(list)

    err = False

    for i in tqdm(list(range(degradation_steps))):
        if i % int(1 / report_frequency) == 0:
            for metric_name, metric in [
                ("muc", score_muc),
                ("b_cubed", score_b_cubed),
                ("ceaf", score_ceaf),
                ("blanc", score_blanc),
                ("lea", score_lea),
                ("mention_detection", score_mention_detection),
            ]:
                p, r, f1 = metric([corefs_pred], [corefs_refs])
                coref_metrics_dict[f"{metric_name}_precision"].append(p)
                coref_metrics_dict[f"{metric_name}_recall"].append(r)
                coref_metrics_dict[f"{metric_name}_f1"].append(f1)

            # caveat: normally preds are split...
            state.corefs = corefs_pred.coref_chains  # type: ignore

            try:
                state = pipeline.rerun_from(state, cu_step.__class__)
            except Exception as e:
                if not err:
                    print(e)
                    err = True
                for k in graph_metrics_dict.keys():
                    graph_metrics_dict[k].append(0)
                continue

            assert not state.characters is None
            assert not state_gold.characters is None

            node_p, node_r, node_f1 = score_character_unification(
                [character.names for character in state_gold.characters],
                [character.names for character in state.characters],
            )
            graph_metrics_dict["node_f1"].append(node_f1)
            graph_metrics_dict["node_precision"].append(node_p)
            graph_metrics_dict["node_recall"].append(node_r)

            mapping, _ = align_characters(state_gold.characters, state.characters)
            edge_p, edge_r, edge_f1 = score_network_extraction_edges(
                state_gold.character_network,
                state.character_network,
                mapping,
                weighted=False,
            )
            graph_metrics_dict["edge_f1"].append(edge_f1)
            graph_metrics_dict["edge_precision"].append(edge_p)
            graph_metrics_dict["edge_recall"].append(edge_r)
            wedge_p, wedge_r, wedge_f1 = score_network_extraction_edges(
                state_gold.character_network,
                state.character_network,
                mapping,
                weighted=True,
            )
            graph_metrics_dict["weighted_edge_f1"].append(wedge_f1)
            graph_metrics_dict["weighted_edge_precision"].append(wedge_p)
            graph_metrics_dict["weighted_edge_recall"].append(wedge_r)

        corefs_pred = deteriorate_coref(
            corefs_pred, corefs_refs, actions=degradation_actions
        )
        if corefs_pred is None:
            measurement_steps = int(degradation_steps * report_frequency)
            current_measurement_steps = len(list(graph_metrics_dict.values())[0])
            remaining_measurement_steps = int(
                measurement_steps - current_measurement_steps
            )
            for k, v in graph_metrics_dict.items():
                graph_metrics_dict[k] += [v[-1]] * remaining_measurement_steps
            for k, v in coref_metrics_dict.items():
                coref_metrics_dict[k] += [v[-1]] * remaining_measurement_steps
            break

    return coref_metrics_dict, graph_metrics_dict


def compute_metrics_over_ner_degradation(
    state_gold: PipelineState,
    character_unification_step: Literal["naive", "rules"],
    character_unification_step_kwargs: dict,
    report_frequency: float,
    degradation_steps: int,
    co_occurrences_dist: Tuple[int, Literal["tokens", "sentences"]],
    degradation_actions: Optional[list[Callable]] = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    :return: ner_metrics_dict, graph_metrics_dict
    """
    cu_step = (
        GraphRulesCharacterUnifier(**character_unification_step_kwargs)
        if character_unification_step == "rules"
        else SpliceNaiveCharacterUnifier(**character_unification_step_kwargs)
    )

    # as in xp.py
    # NOTE: SpliceNaiveCharacterUnifier reproduces the process of
    # extracting a character from entities+corefs as done when loading
    # data.
    pipeline = Pipeline(
        [
            cu_step,
            CoOccurrencesGraphExtractor(co_occurrences_dist=co_occurrences_dist),
        ],
        progress_report=None,
    )
    # necessary since we only use pipeline.rerun_from
    pipeline._pipeline_init_steps_()

    state = deepcopy(state_gold)
    assert not state.tokens is None
    assert not state.entities is None
    assert not state_gold.entities is None

    ner_pred = deepcopy(state.entities)
    ner_ref = deepcopy(state_gold.entities)

    ner_metrics_dict = defaultdict(list)
    graph_metrics_dict = defaultdict(list)

    def store_metrics_(state: PipelineState):
        pred_tags = entities_to_BIO(state.tokens, state.entities)
        ref_tags = entities_to_BIO(state.tokens, ner_ref)

        # seqeval returns 0 or 1 for undefined recall/precision
        # depending on the configuration. However, we would like
        # to explicitely mark these undefined values: therefore,
        # in the case of undefined metrics, we report NaN instead.
        if len(state.entities) == 0:
            precision = float("nan")
        else:
            precision = precision_score([ref_tags], [pred_tags])
        ner_metrics_dict["precision"].append(precision)

        if len(ner_ref) == 0:
            recall = float("nan")
        else:
            recall = recall_score([ref_tags], [pred_tags])
        ner_metrics_dict["recall"].append(recall)

        node_p, node_r, node_f1 = score_character_unification(
            [character.names for character in state_gold.characters],
            [character.names for character in state.characters],
        )
        graph_metrics_dict["node_f1"].append(node_f1)
        graph_metrics_dict["node_precision"].append(node_p)
        graph_metrics_dict["node_recall"].append(node_r)

        mapping, _ = align_characters(state_gold.characters, state.characters)
        edge_p, edge_r, edge_f1 = score_network_extraction_edges(
            state_gold.character_network,
            state.character_network,
            mapping,
            weighted=False,
        )
        graph_metrics_dict["edge_f1"].append(edge_f1)
        graph_metrics_dict["edge_precision"].append(edge_p)
        graph_metrics_dict["edge_recall"].append(edge_r)
        wedge_p, wedge_r, wedge_f1 = score_network_extraction_edges(
            state_gold.character_network,
            state.character_network,
            mapping,
            weighted=True,
        )
        graph_metrics_dict["weighted_edge_f1"].append(wedge_f1)
        graph_metrics_dict["weighted_edge_precision"].append(wedge_p)
        graph_metrics_dict["weighted_edge_recall"].append(wedge_r)

    def try_run_() -> PipelineState:
        try:
            new_state = pipeline.rerun_from(state, cu_step.__class__)
        except Exception as e:
            print(e)
            for k in graph_metrics_dict.keys():
                graph_metrics_dict[k].append(0)
            new_state = None
        return new_state

    store_metrics_(state)

    for i in tqdm(list(range(degradation_steps))):

        old_ner_pred = deepcopy(ner_pred)
        ner_pred = deteriorate_ner(
            state.tokens, ner_pred, ner_ref, actions=degradation_actions
        )
        if ner_pred is None:
            state.entities = old_ner_pred
            state = try_run_()
            if not state is None:
                store_metrics_(state)
            measurement_steps = int(degradation_steps * report_frequency)
            current_measurement_steps = len(list(graph_metrics_dict.values())[0])
            remaining_measurement_steps = int(
                measurement_steps - current_measurement_steps
            )
            for k, v in graph_metrics_dict.items():
                graph_metrics_dict[k] += [v[-1]] * remaining_measurement_steps
            for k, v in ner_metrics_dict.items():
                ner_metrics_dict[k] += [v[-1]] * remaining_measurement_steps
            break

        if i % int(1 / report_frequency) == 0:
            state.entities = ner_pred
            state = try_run_()
            if not state is None:
                store_metrics_(state)

    return ner_metrics_dict, graph_metrics_dict


def plot_task_and_graph_metrics_corr(
    task_metrics_dict: dict[str, list[float]],
    graph_metrics_dict: dict[str, list[float]],
) -> plt.Figure:
    corr_matrix = np.zeros((len(graph_metrics_dict), len(task_metrics_dict)))
    for i, graph_metric in enumerate(graph_metrics_dict.keys()):
        for j, coref_metric in enumerate(task_metrics_dict.keys()):
            corr = np.corrcoef(
                graph_metrics_dict[graph_metric], task_metrics_dict[coref_metric]
            )  # type: ignore
            corr = corr[0][1]
            corr_matrix[i][j] = corr

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1)
    ax.set_xticks(range(len(task_metrics_dict)), task_metrics_dict.keys(), rotation=70)
    ax.set_yticks(range(len(graph_metrics_dict)), graph_metrics_dict.keys())
    plt.tight_layout()
    fig.colorbar(im)

    return fig


def plot_task_and_graph_metrics(
    task_metrics_dict: dict[str, list[float]],
    task_name: str,
    graph_metrics_dict: dict[str, list[float]],
    report_frequency: float = 1.0,
    steps: Optional[int] = None,
) -> plt.Figure:
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(17, 5)

    if steps is None:
        assert report_frequency == 1.0
        steps = len(list(task_metrics_dict.values())[0])

    x = [int(x / report_frequency) for x in range(int(steps * report_frequency))]

    for metric, values in task_metrics_dict.items():
        axs[0].plot(x, values, label=metric)
        axs[0].set_title(f"{task_name} metrics")
    axs[0].legend()

    for metric, values in graph_metrics_dict.items():
        axs[1].plot(x, values, label=metric)
        axs[1].set_title("Graph metrics")
    axs[1].legend()

    return fig
