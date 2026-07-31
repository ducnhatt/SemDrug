# SemDrug: Quality-Aware Knowledge Graph Retrieval and Semantic Proxy Augmentation for Grounded LLM-Based Drug Repurposing

**SemDrug** is a Knowledge Graph-enhanced Large Language Model (LLM) reasoning framework for inductive biomedical relation prediction. While combining LLMs with Knowledge Graphs (KGs) is a powerful paradigm for biomedical reasoning, its effectiveness depends heavily on the quality and coverage of the retrieved KG evidence. 

SemDrug addresses the *retrieval-quality gap* and *KG coverage gap* by improving the retrieval layer through two core components:
1. **Quality-Aware Semantic Maximal Marginal Relevance (MMR) Path Retrieval**: Scores candidate multi-hop paths by semantic relevance, filters weak paths, and selects diverse evidence to prevent diluting the LLM's context window.
2. **Cold-start Semantic Proxy Augmentation**: Addresses KG coverage limitations by connecting isolated endpoint drugs to semantically similar KG-covered proxies via synthetic edges.

![Overview of the SemDrug framework](assets/fig_overview.png)

*Fig. 1. Overview of the SemDrug framework: (1) Data Preprocessing maps queries to KG nodes; (2) KG Construction integrates datasets into an augmented KG; (3) Node Text Construction and Semantic Embedding converts node metadata into vectors; (4) Leakage-safe Graph Preparation removes direct edges and applies semantic proxy augmentation; (5) Quality-Aware KG Retrieval extracts and filters multi-hop paths; and (6) LLM-based Inference linearizes paths into textual prompts for zero-shot LLM prediction.*

---

## Supported Tasks

The project currently supports experiments on two primary biomedical benchmarks:

* **PharmacotherapyDB**: drug-disease therapeutic indication prediction
* **DDInter**: drug-drug interaction severity prediction

The main thesis experiments focus on **PharmacotherapyDB** under an inductive drug-disease therapeutic indication prediction setting, and **DDInter** under a cold-start setting.

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


