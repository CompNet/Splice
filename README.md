# Splice

# The Role of Information Extraction Tasks in Automatic Literary Character Network Construction

## Reproducing Results

First, you should:

- install dependencies. Either use `poetry install` if you have poetry, or `pip install -r requirements.txt` otherwise.
- get the [litbank dataset](https://github.com/dbamman/litbank)

The main experiment can be run with `xp.py`:

```sh
python xp.py with\
	   min_graph_nodes=10\
	   co_occurrences_dist=32\
	   litbank.root="/path/to/litbank"
```


### Degradation Experiments

The following script will run all of the degradation experiments:

```sh
MAIN_XP_RUN="/path/to/main/xp/run"

python xp_metrics_over_degradation.py with input_dir="${MAIN_XP_RUN}" task_name=NER degradation_name=add_wrong_entity degradation_steps=1000 degradation_report_frequency=0.05
python xp_metrics_over_degradation.py with input_dir="${MAIN_XP_RUN}" task_name=NER degradation_name=remove_correct_entity degradation_steps=200 degradation_report_frequency=0.5
python xp_metrics_over_degradation.py with input_dir="${MAIN_XP_RUN}" task_name=coref degradation_name=add_wrong_mention degradation_steps=200 degradation_report_frequency=0.05
python xp_metrics_over_degradation.py with input_dir="${MAIN_XP_RUN}" task_name=coref degradation_name=remove_correct_mention degradation_steps=1000 degradation_report_frequency=0.05
python xp_metrics_over_degradation.py with input_dir="${MAIN_XP_RUN}" task_name=coref degradation_name=add_wrong_link degradation_steps=500 degradation_report_frequency=0.05
python xp_metrics_over_degradation.py with input_dir="${MAIN_XP_RUN}" task_name=coref degradation_name=remove_correct_link degradation_steps=1000 degradation_report_frequency=0.05
python xp_metrics_over_degradation.py with input_dir="${MAIN_XP_RUN}" task_name=coref degradation_name=coref_all degradation_steps=1000 degradation_report_frequency=0.05
```


### End-to-end LLM-based Pipelines

The *E2E-Coref* experiment can be reproduced with the `xp_e2e_llm_coref.py` script:

```sh
MAIN_XP_RUN="/path/to/main/xp/run"
LITBANK_PATH="/path/to/litbank"

python xp_e2e_llm_coref.py with\
	   input_dir="${MAIN_XP_RUN}"\
	   model="gpt3.5"\
	   openAI_API_key="insert your openAI key"\
	   litbank.root="${LITBANK_PATH}"

python xp_e2e_llm_coref.py with\
	   input_dir="${MAIN_XP_RUN}"\
	   model="gpt40"\
	   openAI_API_key="insert your openAI key"\
	   litbank.root="${LITBANK_PATH}"

python xp_e2e_llm_coref.py with\
	   input_dir="${MAIN_XP_RUN}"\
	   model="llama3-8b-instruct"\
	   hg_access_token="insert your Huggingface access token"\
	   device="cuda"\
	   litbank.root="${LITBANK_PATH}"
```

Similarly, the *E2E-Graphml experiment can be reproduced with the `xp_e2e_llm_graphml.py` script:

```sh
MAIN_XP_RUN="/path/to/main/xp/run"

python xp_e2e_llm_graphml.py with\
	   input_dir="${MAIN_XP_RUN}"\
	   model="gpt3.5"\
	   openAI_API_key="insert your openAI key"\
	   litbank.root="${LITBANK_PATH}"

python xp_e2e_llm_graphml.py with\
	   input_dir="${MAIN_XP_RUN}"\
	   model="gpt40"\
	   openAI_API_key="insert your openAI key"\
	   litbank.root="${LITBANK_PATH}"

python xp_e2e_llm_graphml.py with\
	   input_dir="${MAIN_XP_RUN}"\
	   model="llama3-8b-instruct"\
	   hg_access_token="insert your Huggingface access token"\
	   device="cuda"\
	   litbank.root="${LITBANK_PATH}"
```


### Printing / Plotting Results

| Figure   | Corresponding Script                |
|----------|-------------------------------------|
| Table 1  | `print_main_task_results.py`        |
| Table 2  | `print_main_graph_results.py`       |
| Table 3  |                                     |
| Figure 1 | `plot_degradation_metrics.py`       |
| Figure 2 | `plot_ner_degradation_metrics.py`   |
| Figure 3 | `plot_coref_degradation_metrics.py` |
| Table 4  | `print_e2e_graph_results.py`        |
