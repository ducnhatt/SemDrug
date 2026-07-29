
# SemDrug

**SemDrug** is a Knowledge Graph-enhanced Large Language Model framework for drug repurposing and biomedical relation reasoning. The project focuses on improving the quality of retrieved biomedical Knowledge Graph evidence before it is passed to an LLM for final prediction.

The core method is **Quality-Constrained Semantic Maximal Marginal Relevance Path Retrieval**, which retrieves multi-hop biomedical paths that are semantically relevant, quality-controlled, and diverse.

---

## Main Idea

Large Language Models can reason over biomedical questions, but they may hallucinate when no grounded evidence is provided. Biomedical Knowledge Graphs provide structured evidence, but raw graph paths can be noisy, generic, or redundant.

SemDrug addresses this by inserting a retrieval layer between the Knowledge Graph and the LLM:

```text
Drug-disease or drug-drug query
        ↓
Augmented biomedical Knowledge Graph
        ↓
Leakage-safe multi-hop path retrieval
        ↓
Semantic scoring and quality filtering
        ↓
MMR-based evidence selection
        ↓
Path linearization
        ↓
LLM-based prediction
````

---

## Supported Tasks

The project currently supports experiments on:

* **PharmacotherapyDB**: drug-disease therapeutic indication prediction
* **DDInter**: drug-drug interaction severity prediction

The main thesis experiments focus on **PharmacotherapyDB** under an inductive drug-disease therapeutic indication prediction setting.

---

## Project Structure

```text
SemDrug/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── semdrug/
│   ├── __init__.py
│   ├── create_augmented_network.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── augmented_network_utils.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── filter_path_embeddings.py
│   │   └── kpaths_utils.py
│   │
│   └── llm/
│       ├── __init__.py
│       ├── llm_inference.py
│       ├── evaluate_llm_regex.py
│       └── evaluate_llm_bertscore.py
│
├── data/
│   ├── hetionet/
│   ├── pharmaDB/
│   ├── ddinter/
│   ├── drugbank/
│   ├── paths/
│   └── embeddings/
│
├── outputs/
│   ├── pharmaDB/
│   ├── ddinter/
│   └── drugbank/
│
├── prompts/
├── scripts/
└── notebooks/
```

---

## Environment Setup

Create a Python virtual environment:

```powershell
python -m venv .semdrug-env
.\.semdrug-env\Scripts\activate
```

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install the main packages manually:

```powershell
pip install pandas numpy networkx scikit-learn tqdm datasets sentence-transformers torch transformers vllm python-dotenv
```

---

## Hugging Face Authentication

Some LLaMA models require Hugging Face authentication.

Create a `.env` file in the project root:

```text
HF_API_KEY=your_huggingface_token_here
```

Do **not** commit `.env` to Git.

The source code should load the token like this:

```python
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY:
    login(token=HF_API_KEY)
```

---

## `.gitignore`

Recommended `.gitignore`:

```text
__pycache__/
*.pyc

.env
.semdrug-env/
.venv/

outputs/
*.csv
*.pkl
*.pt
*.bin
*.safetensors

.DS_Store
.ipynb_checkpoints/
```

---

## Data Preparation

The expected input files are stored under the `data/` directory.

For each dataset, SemDrug expects:

```text
node2id.json
BKG_file.txt
training file
relation dictionary
```

Example for PharmacotherapyDB:

```text
data/pharmaDB/node2id.json
data/pharmaDB/BKG_file.txt
data/pharmaDB/pharmaDB_train.txt
data/pharmaDB/pharmaDB_relations.json
```

Hetionet relation files are also required:

```text
data/hetionet/hetionet_relations.json
data/hetionet/hetionet_reversed_relations.json
```

---

## Step 1: Build Augmented Knowledge Graphs

Run:

```powershell
python -m semdrug.create_augmented_network
```

This creates augmented KG files such as:

```text
data/pharmaDB/pharmaDB_Augmented_KG.txt
data/ddinter/ddinter_Augmented_KG.txt
data/relations_dicts_file_all.json
```

The augmented KG combines:

```text
Base biomedical KG + training-set task relations
```

For evaluation queries, direct query edges are removed later to prevent label leakage.

---

## Step 2: Retrieve Knowledge Graph Paths

Run semantic-MMR path retrieval.

Example for PharmacotherapyDB:

```powershell
python -m semdrug.retrieval.filter_path_embeddings ^
  --dataset_name pharmaDB ^
  --add_reverse_edges ^
  --candidate_pool 32 ^
  --max_final_paths 10 ^
  --mmr_lambda 0.5 ^
  --score_floor_delta 0.05
```

Example for DDInter:

```powershell
python -m semdrug.retrieval.filter_path_embeddings ^
  --dataset_name ddinter ^
  --add_reverse_edges ^
  --candidate_pool 64 ^
  --max_final_paths 10 ^
  --mmr_lambda 0.5 ^
  --score_floor_delta 0.05
```

The output is a JSON file containing retrieved paths, for example:

```text
outputs/pharmaDB_semantic_mmr_d3_pool32_k10_mmr0.5_delta0.05.json
```

Each example may include fields such as:

```text
path_str
all_paths
path_count
raw_path_count
retrieval_source
complete_evidence
used_leakage_safety_recovery
```

---

## Step 3: Run LLM Inference

Example for PharmacotherapyDB with KG evidence:

```powershell
python -m semdrug.llm.llm_inference ^
  --dataset_name pharmacotherapydb ^
  --dataset_path outputs/pharmaDB_semantic_mmr_d3_pool32_k10_mmr0.5_delta0.05.json ^
  --use_kg ^
  --use_options ^
  --max_tokens 32 ^
  --output_dir outputs/pharmaDB
```

Example for DDInter:

```powershell
python -m semdrug.llm.llm_inference ^
  --dataset_name ddinter ^
  --dataset_path outputs/ddinter_semantic_mmr_d3_pool64_k10_mmr0.5_delta0.05.json ^
  --use_kg ^
  --use_options ^
  --max_tokens 8 ^
  --output_dir outputs/ddinter
```

The prediction file is saved as:

```text
outputs/<run_name>/predictions.csv
```

---

## Step 4: Evaluate Predictions

Example for PharmacotherapyDB:

```powershell
python -m semdrug.llm.evaluate_llm_regex ^
  --dataset pharmaDB ^
  --prediction_path outputs/pharmaDB/<run_name>/predictions.csv ^
  --use_options
```

Example for DDInter:

```powershell
python -m semdrug.llm.evaluate_llm_regex ^
  --dataset ddinter ^
  --prediction_path outputs/ddinter/<run_name>/predictions.csv ^
  --use_options
```

The evaluator reports:

```text
Accuracy
Macro F1-score
Cohen's Kappa
Classification report
Confusion matrix
Prediction label distribution
Parse error rate
```

---

## Main Retrieval Arguments

| Argument                     | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `--dataset_name`             | Dataset name: `pharmaDB`, `ddinter`, or `drugbank`   |
| `--add_reverse_edges`        | Adds inverse relations to improve graph traversal    |
| `--max_depth`                | Maximum path length                                  |
| `--candidate_pool`           | Number of candidate paths generated before reranking |
| `--max_final_paths`          | Number of paths passed to the LLM                    |
| `--mmr_lambda`               | Redundancy penalty strength in MMR selection         |
| `--score_floor_delta`        | Quality margin for filtering weak candidate paths    |
| `--disable_partial_fallback` | Disables weak partial-neighbor fallback              |
| `--disable_fallback`         | Disables complete bridge fallback                    |

---

## Main LLM Arguments

| Argument                  | Description                                 |
| ------------------------- | ------------------------------------------- |
| `--dataset_name`          | Dataset name                                |
| `--dataset_path`          | Path to retrieval JSON file                 |
| `--model_name_or_path`    | Hugging Face model name or local model path |
| `--use_kg`                | Include retrieved KG paths in the prompt    |
| `--use_drug_descriptions` | Include drug/disease descriptions           |
| `--use_options`           | Include label options and definitions       |
| `--max_tokens`            | Maximum generated tokens                    |
| `--temperature`           | Decoding temperature                        |
| `--output_dir`            | Directory for predictions                   |

---

## Reproducibility Notes

* All primary experiments use greedy decoding with `temperature=0.0`.
* Query-specific direct edges are removed before path retrieval to prevent label leakage.
* Retrieved paths are filtered, diversified, and then reordered by semantic score before LLM prompting.
* The final result may depend on the exact model checkpoint, prompt template, package versions, and GPU environment.
* Do not compare runs unless they use the same dataset split, prompt format, model, decoding settings, and evaluator.

---

## Current Best Thesis Result

On PharmacotherapyDB under the inductive setting, the final SemDrug configuration achieves:

```text
Accuracy:       72.62%
Macro F1-score: 71.40%
Cohen's Kappa:  58.20%
```

These results show that improving the retrieval layer can improve KG-enhanced LLM reasoning without fine-tuning the LLM.

---

## Important Security Note

Do not hardcode Hugging Face tokens or API keys inside source code.

Use `.env` instead:

```text
HF_API_KEY=your_huggingface_token_here
```

and keep `.env` in `.gitignore`.

---

## TODO

* Replace remaining hardcoded Kaggle paths with configurable local paths.
* Add YAML configuration files for each dataset.
* Add shell/PowerShell scripts for common experiment runs.
* Cleanly separate DrugBank-specific, DDInter-specific, and PharmacotherapyDB-specific prompt logic.
* Add unit tests for path filtering and leakage prevention.
* Improve documentation for data preparation and expected file formats.

---

## License

This project is developed for undergraduate thesis research. Please add an appropriate license before public release.

```
```
