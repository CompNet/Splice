from __future__ import annotations
import json, pickle
import pathlib as pl
import functools as ft
import matplotlib.pyplot as plt
from sacred import Experiment
from sacred.run import Run
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from splice.analysis import (
    compute_metrics_over_ner_degradation,
    compute_metrics_over_coref_degradation,
    plot_task_and_graph_metrics,
)
from splice.corefs import (
    add_wrong_link,
    add_wrong_mention,
    deteriorate_coref,
    remove_correct_link,
    remove_correct_mention,
)
from splice.ner import add_wrong_entity, remove_correct_entity
from splice.sacred_utils import archive_fig, log_series
from splice.utils import mean_noNaN

DEGRADATION_ACTIONS = {
    "add_wrong_entity": ft.partial(add_wrong_entity, max_size=3),
    "remove_correct_entity": remove_correct_entity,
    "add_wrong_mention": ft.partial(add_wrong_mention, max_span_size=10),
    "remove_correct_mention": remove_correct_mention,
    "add_wrong_link": add_wrong_link,
    "remove_correct_link": remove_correct_link,
    "coref_all": ft.partial(
        deteriorate_coref,
        actions=[
            ft.partial(add_wrong_mention, max_span_size=10),
            remove_correct_mention,
            add_wrong_link,
            remove_correct_link,
        ],
    ),
}

ex = Experiment("metrics_over_degradation")
ex.observers.append(FileStorageObserver("./runs"))


@ex.config
def config():
    input_dir: str
    task_name: str
    degradation_name: str
    degradation_steps: int
    degradation_report_frequency: float


@ex.automain
def main(
    _run: Run,
    input_dir: str,
    task_name: str,
    degradation_name: str,
    degradation_steps: int,
    degradation_report_frequency: float,
):
    print_config(_run)

    RUN_PATH = pl.Path(input_dir)

    with open(RUN_PATH / "info.json") as f:
        info = json.load(f)

    with open(RUN_PATH / "run.json") as f:
        run_dict = json.load(f)
    co_occurrences_dist = run_dict["meta"]["config_updates"]["co_occurrences_dist"]

    analysis_novels = info["analysis_novels"]

    book_states = {}
    book_states_gold = {}
    for book in analysis_novels:
        with open(RUN_PATH / f"{book}_state.pickle", "rb") as f:
            book_states[book] = pickle.load(f)
        with open(RUN_PATH / f"{book}_state_gold.pickle", "rb") as f:
            book_states_gold[book] = pickle.load(f)

    print(
        f"starting evaluation of {task_name} task under {degradation_name} degradation"
    )

    all_task_metrics_dict: list[dict[str, list[float]]] = []
    all_graph_metrics_dict: list[dict[str, list[float]]] = []

    degradation_action = DEGRADATION_ACTIONS[degradation_name]

    for book, book_state_gold in book_states_gold.items():
        if task_name == "NER":
            (
                task_metrics_dict,
                graph_metrics_dict,
            ) = compute_metrics_over_ner_degradation(
                book_state_gold,
                "rules",
                {"link_corefs_mentions": True},
                degradation_report_frequency,
                degradation_steps,
                (co_occurrences_dist, "tokens"),
                degradation_actions=[degradation_action],
            )

        elif task_name == "coref":
            (
                task_metrics_dict,
                graph_metrics_dict,
            ) = compute_metrics_over_coref_degradation(
                book_state_gold,
                "rules",
                {"link_corefs_mentions": True},
                degradation_report_frequency,
                degradation_steps,
                (co_occurrences_dist, "tokens"),
                degradation_actions=[degradation_action],
            )

        else:
            raise ValueError()

        for metric, values in task_metrics_dict.items():
            steps = [int(x / degradation_report_frequency) for x in range(len(values))]
            log_series(_run, f"{book}_{metric}", values, steps)
        for metric, values in graph_metrics_dict.items():
            steps = [int(x / degradation_report_frequency) for x in range(len(values))]
            log_series(_run, f"{book}_{metric}", values, steps)

        fig = plot_task_and_graph_metrics(
            task_metrics_dict,
            task_name,
            graph_metrics_dict,
            report_frequency=degradation_report_frequency,
            steps=degradation_steps,
        )
        archive_fig(_run, fig, book)

        all_task_metrics_dict.append(task_metrics_dict)
        all_graph_metrics_dict.append(graph_metrics_dict)

    task_metrics_keys = all_task_metrics_dict[0].keys()
    mean_task_metrics_dict = {
        k: [
            mean_noNaN([d[k][i] for d in all_task_metrics_dict])
            for i in range(len(all_task_metrics_dict[0][k]))
        ]
        for k in task_metrics_keys
    }

    graph_metrics_keys = all_graph_metrics_dict[0].keys()
    mean_graph_metrics_dict = {
        k: [
            mean_noNaN([d[k][i] for d in all_graph_metrics_dict])
            for i in range(len(all_graph_metrics_dict[0][k]))
        ]
        for k in graph_metrics_keys
    }

    for metric, values in mean_task_metrics_dict.items():
        steps = [int(x / degradation_report_frequency) for x in range(len(values))]
        log_series(_run, f"mean_{metric}", values, steps)
    for metric, values in mean_graph_metrics_dict.items():
        steps = [int(x / degradation_report_frequency) for x in range(len(values))]
        log_series(_run, f"mean_{metric}", values, steps)

    fig = plot_task_and_graph_metrics(
        mean_task_metrics_dict,
        task_name,
        mean_graph_metrics_dict,
        report_frequency=degradation_report_frequency,
        steps=degradation_steps,
    )
    plt.tight_layout()
    archive_fig(_run, fig, "MEAN")
