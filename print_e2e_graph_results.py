import argparse, json
import pathlib as pl
import pandas as pd

if __name__ == "__main__":
    llm_runsets = {
        "e2ecoref": {
            "Llama3-8b-instruct": "./runs/llama3-8b-instruct_e2ecoref",
            "GPT-3.5 Turbo": "./runs/gpt3.5_e2ecoref",
            "GPT4o": "./runs/gpt4o_e2ecoref",
        },
        "e2egraphml": {
            "Llama3-8b-instruct": "./runs/llama3-8b-instruct_e2egraphml",
            "GPT-3.5 Turbo": "./runs/gpt3.5_e2egraphml",
            "GPT4o": "./runs/gpt4o_e2egraphml",
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runset",
        type=str,
        default="e2ecoref",
        help=f"one of {list(llm_runsets.keys())}",
    )
    args = parser.parse_args()

    runs = llm_runsets[args.runset]

    metrics_dict = {}
    for model_key, run_dir in runs.items():
        with open(pl.Path(run_dir) / "metrics.json") as f:
            metrics_dict[model_key] = json.load(f)

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

    def get_value(model: str, metric: str) -> float:
        try:
            raw_value = metrics_dict[model][f"MEAN_{metric}"]["values"][0]
        except KeyError:
            raw_value = metrics_dict[model][f"MEAN_pipeline.{metric}"]["values"][0]
        return round(raw_value * 100, 2)

    df_dict = {
        model: [get_value(model, metric) for metric in METRICS_TO_PRINT]
        for model in runs.keys()
    }
    df_dict["Metric"] = METRICS_TO_PRINT

    df = pd.DataFrame(df_dict)
    df = df.set_index("Metric")
    print(df)
