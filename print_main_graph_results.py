import json, argparse
import pathlib as pl
from typing import Literal
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=str)
    args = parser.parse_args()

    with open(pl.Path(args.run) / "metrics.json") as f:
        metrics_dict = json.load(f)

    METRICS_TO_PRINT = [
        "node_f1",
        "node_precision",
        "node_recall",
        "edge_f1",
        "edge_precision",
        "edge_recall",
        "weighted_edge_f1",
        "weighted_edge_precision",
        "weighted_edge_recall",
    ]

    def get_value(metric: str, pipeline: Literal["coref", "nocoref"]) -> float:
        pipeline_name = "pipeline" if pipeline == "coref" else "pipeline_nocoref"
        return round(
            metrics_dict[f"MEAN_{pipeline_name}.{metric}"]["values"][0] * 100, 2
        )

    df = pd.DataFrame(
        {
            "Metric": METRICS_TO_PRINT,
            "w/ coreference": [
                get_value(metric, "coref") for metric in METRICS_TO_PRINT
            ],
            "w/o coreference": [
                get_value(metric, "nocoref") for metric in METRICS_TO_PRINT
            ],
        }
    )
    df = df.set_index("Metric")
    print(df)
