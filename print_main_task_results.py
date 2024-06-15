import json, argparse
import pathlib as pl
from typing import Literal
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=str)
    args = parser.parse_args()

    run_path = pl.Path(args.run)

    with open(run_path / "metrics.json") as f:
        metrics_dict = json.load(f)

    with open(run_path / "info.json") as f:
        info = json.load(f)
    novels = info["analysis_novels"]

    METRICS_TO_PRINT = [
        "ner_f1",
        "MUC_f1",
        "B3_f1",
        "CEAF_f1",
        "BLANC_f1",
        "LEA_f1",
    ]

    def get_mean_value(metric: str) -> float:
        return round(metrics_dict[f"MEAN_{metric}"]["values"][0] * 100, 2)

    def get_min_value(metric: str) -> float:
        return round(
            min([metrics_dict[f"{novel}.{metric}"]["values"][0] for novel in novels])
            * 100,
            2,
        )

    def get_max_value(metric: str) -> float:
        return round(
            max([metrics_dict[f"{novel}.{metric}"]["values"][0] for novel in novels])
            * 100,
            2,
        )

    df = pd.DataFrame(
        {
            "Metric": METRICS_TO_PRINT,
            "Mean": [get_mean_value(metric) for metric in METRICS_TO_PRINT],
            "Min": [get_min_value(metric) for metric in METRICS_TO_PRINT],
            "Max": [get_max_value(metric) for metric in METRICS_TO_PRINT],
        }
    )
    df = df.set_index("Metric")
    print(df)
