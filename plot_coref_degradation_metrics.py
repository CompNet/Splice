from __future__ import annotations
import argparse, json
import pathlib as pl
from more_itertools.recipes import flatten
import scienceplots
import matplotlib.pyplot as plt


def precision_recall_metrics(metrics: set[str]) -> set[str]:
    precision_metrics = {f"{metric}_precision" for metric in metrics}
    recall_metrics = {f"{metric}_recall" for metric in metrics}
    return precision_metrics | recall_metrics


GMETRIC_2_PRETTYNAME = {
    "node_f1": "$F1_V$",
    "node_precision": "$Pre_V$",
    "node_recall": "$Rec_V$",
    "edge_f1": "$F1_E$",
    "edge_precision": "$Pre_E$",
    "edge_recall": "$Rec_E$",
    "weighted_edge_f1": "$WF1_E$",
    "weighted_edge_precision": "$WPre_E$",
    "weighted_edge_recall": "$WRec_E$",
}


def capitalize_snakecase_text(text: str) -> str:
    if text in GMETRIC_2_PRETTYNAME:
        return GMETRIC_2_PRETTYNAME[text]
    if text.startswith("lea_"):
        return "LEA " + capitalize_snakecase_text(text[4:])
    if text.startswith("ceaf_"):
        return "CEAF " + capitalize_snakecase_text(text[5:])
    if text.startswith("b_cubed_"):
        return "$B^3$ " + capitalize_snakecase_text(text[8:])
    if text.startswith("muc_"):
        return "MUC " + capitalize_snakecase_text(text[4:])
    splitted = text.split("_")
    return " ".join([word.capitalize() for word in splitted])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run-dict",
        type=eval,
        default={
            "add_spurious_coref_mention": "./runs/add_spurious_coref_mention/",
            "add_spurious_link": "./runs/add_spurious_link/",
            "remove_correct_coref_mention": "./runs/remove_correct_coref_mention/",
            "remove_correct_link": "./runs/remove_correct_link/",
        },
    )
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    FONTSIZE = 8
    TEXT_WIDTH_IN = 6.29921
    ASPECT_RATIO = 0.6

    plt.style.use(["science", "grid"])
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    cmap = plt.get_cmap("tab20")

    fig, axs = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * ASPECT_RATIO))

    TASK_METRICS = {"muc_f1", "b_cubed_f1", "ceaf_f1", "blanc_f1", "lea_f1"}
    GRAPH_METRICS = precision_recall_metrics({"node", "edge", "weighted_edge"})

    for run_i, (pkey, ppath) in enumerate(args.run_dict.items()):
        ax = list(flatten(axs))[run_i]
        with open(pl.Path(ppath) / "metrics.json") as f:
            metrics_dict = json.load(f)
        for metric_i, metric in enumerate(sorted(TASK_METRICS | GRAPH_METRICS)):
            metrics_key = f"mean_{metric}"
            steps = metrics_dict[metrics_key]["steps"]
            values = metrics_dict[metrics_key]["values"]
            ax.plot(
                steps,
                values,
                linestyle="-" if metric in GRAPH_METRICS else "--",
                label=capitalize_snakecase_text(metric),
                linewidth=1.5,
                color=cmap(metric_i),
            )

        ax.set_title(capitalize_snakecase_text(pkey), fontsize=FONTSIZE)
        ax.set_xlabel("Degradation Steps", fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fancybox=True,
        loc="upper left",
        bbox_to_anchor=(1.0, 0.9),
        fontsize=FONTSIZE,
    )
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"plot saved at {args.output}")
    else:
        plt.show()
