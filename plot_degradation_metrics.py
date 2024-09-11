from __future__ import annotations
import argparse, json
import pathlib as pl
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
    NER_DEGRADATIONS = {"add_wrong_entity", "remove_correct_entity"}

    COREF_DEGRADATIONS = {
        "remove_correct_link",
        "remove_correct_mention",
        "add_wrong_link",
        "add_wrong_mention",
        "coref_all",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=str)
    parser.add_argument(
        "-d",
        "--degradation",
        type=str,
        help=f"one of {NER_DEGRADATIONS | COREF_DEGRADATIONS}",
    )
    parser.add_argument("-m", "--metrics", nargs="+", default=["task", "graph"])
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    FONTSIZE = 10
    TEXT_WIDTH_IN = 6.29921
    ASPECT_RATIO = 0.45

    if args.degradation in NER_DEGRADATIONS:
        TASK_METRICS = {"precision", "recall"}
    elif args.degradation in COREF_DEGRADATIONS:
        TASK_METRICS = {"muc_f1", "b_cubed_f1", "ceaf_f1", "blanc_f1", "lea_f1"}
    else:
        raise ValueError(f"unknown degradation: {args.degradation}")

    GRAPH_METRICS = precision_recall_metrics({"node", "edge", "weighted_edge"})

    if not "task" in args.metrics:
        TASK_METRICS = set()
    if not "graph" in args.metrics:
        GRAPH_METRICS = set()

    with open(pl.Path(args.run) / "metrics.json") as f:
        METRICS_DICT = json.load(f)

    plt.style.use(["science", "grid"])
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * ASPECT_RATIO))

    plots = []
    cmap = plt.get_cmap("tab20")
    for i, metric in enumerate(sorted(TASK_METRICS | GRAPH_METRICS)):
        metrics_key = f"mean_{metric}"
        steps = METRICS_DICT[metrics_key]["steps"]
        values = METRICS_DICT[metrics_key]["values"]
        _plot, *_ = ax.plot(
            steps,
            values,
            linestyle="-" if metric in GRAPH_METRICS else "--",
            label=metric,
            linewidth=1.5,
            color=cmap(i),
        )
        plots.append(_plot)

    ax.set_title(capitalize_snakecase_text(args.degradation), fontsize=FONTSIZE)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        plots,
        [
            capitalize_snakecase_text(label)
            for label in sorted(TASK_METRICS | GRAPH_METRICS)
        ],
        fancybox=True,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=FONTSIZE,
    )
    ax.set_xlabel("Degradation Steps")

    if args.output:
        plt.savefig(args.output)
        print(f"plot saved at {args.output}")
    else:
        plt.show()
