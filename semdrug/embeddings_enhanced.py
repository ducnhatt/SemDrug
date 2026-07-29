import os
import re
import json
import pickle
import argparse
import logging
from collections import Counter
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


PROJECT_ROOT = os.environ.get("SEMDRUG_ROOT", "/workspace/SemDrug")
DATA_ROOT = os.environ.get("SEMDRUG_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))

PATHS = {
    "drugbank": {
        "bkg_entity2id": os.path.join(DATA_ROOT, "drugbank", "BKG_entity2Id.json"),
        "node2id": os.path.join(DATA_ROOT, "drugbank", "node2id.json"),
        "augmented_kg": os.path.join(DATA_ROOT, "drugbank", "drugbank_Augmented_KG.txt"),
        "hetionet": os.path.join(DATA_ROOT, "hetionet", "hetionet-v1.0.json"),
        "drug_info": os.path.join(DATA_ROOT, "drugbank", "id_to_name_mapping.json"),
        "train_set": os.path.join(DATA_ROOT, "drugbank", "drugbank_train_set.csv"),
        "test_set": os.path.join(DATA_ROOT, "drugbank", "drugbank_test_set.json"),
    },
    "ddinter": {
        "bkg_entity2id": os.path.join(DATA_ROOT, "ddinter", "BKG_entity2Id.json"),
        "node2id": os.path.join(DATA_ROOT, "ddinter", "node2id.json"),
        "augmented_kg": os.path.join(DATA_ROOT, "ddinter", "ddinter_Augmented_KG.txt"),
        "hetionet": os.path.join(DATA_ROOT, "hetionet", "hetionet-v1.0.json"),
        "drug_info": os.path.join(DATA_ROOT, "ddinter", "id_to_name_mapping.csv"),
        "train_set": os.path.join(DATA_ROOT, "ddinter", "ddinter_train_set.csv"),
        "test_set": os.path.join(DATA_ROOT, "ddinter", "ddinter_test_set.json"),
    },
    "pharmaDB": {
        "bkg_entity2id": os.path.join(DATA_ROOT, "pharmaDB", "BKG_entity2Id.json"),
        "node2id": os.path.join(DATA_ROOT, "pharmaDB", "node2id.json"),
        "augmented_kg": os.path.join(DATA_ROOT, "pharmaDB", "pharmaDB_Augmented_KG.txt"),
        "hetionet": os.path.join(DATA_ROOT, "hetionet", "hetionet-v1.0.json"),
        "drug_info": os.path.join(DATA_ROOT, "pharmaDB", "id_to_name_mapping.json"),
        "train_set": os.path.join(DATA_ROOT, "pharmaDB", "pharmaDB_train_set.csv"),
        "test_set": os.path.join(DATA_ROOT, "pharmaDB", "pharmaDB_test_set.json"),
    },
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_or_jsonl(path: str) -> pd.DataFrame:

    with open(path, "r", encoding="utf-8") as f:
        first_nonspace = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_nonspace = ch
                break

    if first_nonspace in {"[", "{"}:
        try:
            return pd.read_json(path)
        except ValueError:
            pass

    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)


def read_table_auto(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext in {".json", ".jsonl"}:
        return read_json_or_jsonl(path)

    if ext == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def safe_str(x: Any) -> str:
    if x is None:
        return ""

    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    s = str(x).strip()

    if s.lower() in {"nan", "none", "null"}:
        return ""

    return s


def safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        return bool(x)

    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes"}

    return False


def clean_text(x: Any) -> str:
    text = safe_str(x)
    text = text.replace("::", " ")
    text = text.replace("_", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0]


def load_raw_kg_nodes(kg_path: str) -> Tuple[Set[int], Counter]:
    kg_nodes: Set[int] = set()
    occurrences = Counter()

    with open(kg_path, "r", encoding="utf-8") as f:
        first = f.readline()
        parts = first.strip().split()

        if len(parts) == 3:
            try:
                a, b, _ = map(int, parts)
                kg_nodes.add(a)
                kg_nodes.add(b)
                occurrences[a] += 1
                occurrences[b] += 1
            except ValueError:
                pass

        for line in f:
            parts = line.strip().split()

            if len(parts) != 3:
                continue

            try:
                a, b, _ = map(int, parts)
            except ValueError:
                continue

            kg_nodes.add(a)
            kg_nodes.add(b)
            occurrences[a] += 1
            occurrences[b] += 1

    return kg_nodes, occurrences

#Dict mapping Hetionet bkg "kind::identifier" -> "Kind: Name. ID: . Description"
def build_hetionet_key_to_text(hetionet_path: str) -> Dict[str, str]:
    data = load_json(hetionet_path)
    nodes = data.get("nodes", [])

    key_to_text = {}

    resolved_with_description = 0
    resolved_name_only = 0
    fallback_identifier_only = 0

    for node in nodes:
        kind = safe_str(node.get("kind"))
        identifier = safe_str(node.get("identifier"))
        name = safe_str(node.get("name"))

        node_data = node.get("data", {})
        description = ""

        if isinstance(node_data, dict):
            description = safe_str(node_data.get("description", ""))

        if not kind or not identifier:
            continue

        key = f"{kind}::{identifier}"

        if name and description:
            key_to_text[key] = clean_text(f"{kind}: {name}. ID: {key}. {description}")
            resolved_with_description += 1

        elif name:
            key_to_text[key] = clean_text(f"{kind}: {name}. ID: {key}")
            resolved_name_only += 1

        else:
            key_to_text[key] = clean_text(f"{kind}: {identifier}")
            fallback_identifier_only += 1

    logger.info(f"Hetionet readable mappings: {len(key_to_text)}")
    logger.info(f"Hetionet nodes with name + description: {resolved_with_description}")
    logger.info(f"Hetionet nodes with name only: {resolved_name_only}")
    logger.info(f"Hetionet nodes fallback to identifier only: {fallback_identifier_only}")

    return key_to_text

def infer_entity_kind(entity_id: Any) -> str:
    entity_id = safe_str(entity_id)

    if entity_id.startswith("DB"):
        return "Drug"

    if entity_id.startswith("DOID"):
        return "Disease"

    return "Entity"

# modify drug_texts: dict mapping db -> text
def add_drug_text(
    drug_texts: Dict[str, str],
    drugbank_id: Any,
    name: Any = "",
    desc: Any = "",
    max_chars: int = 1500,
    text_mode: str = "full",
):
    """
    Add DrugBank ID -> "name. description".
    If duplicated, keep the longer/more informative text.
    """
    db = safe_str(drugbank_id)
    name = safe_str(name)
    desc = safe_str(desc)

    if not db:
        return

    if not name and not desc:
        return
    
    if text_mode == "name_only":
        text = name if name else db

    elif text_mode == "short":
        kind = infer_entity_kind(db)
        if name:
            # short_desc = truncate_text(desc, max_chars=max_chars)
            text = f"{kind}: {name}. ID: {db}"
        # elif name:
        #     text = name
        else:
            text = f"{kind}: {db}"

    else:
        if name and desc:
            text = f"{name}. {desc}"
        elif name:
            text = name
        else:
            text = desc

        text = truncate_text(text, max_chars=max_chars)

    text = clean_text(text)

    if db not in drug_texts or len(text) > len(drug_texts[db]):
        drug_texts[db] = text


def collect_drug_texts_from_df(
    df: pd.DataFrame,
    source_name: str,
    drug_texts: Dict[str, str],
    max_chars: int,
    text_mode: str,
):
    """
    Extract drug text from DDInter-like dataframes.
    """
    logger.info(f"Processing drug text source: {source_name}")
    logger.info(f"{source_name} columns: {list(df.columns)}")

    candidate_sets = [
        # DDInter
        ("drug1_db", "drug1_name", "drug1_desc"),
        ("drug2_db", "drug2_name", "drug2_desc"),
        ("Drug1_ID", "drug1_name", "drug1_desc"),
        ("Drug2_ID", "drug2_name", "drug2_desc"),

        # PharmaDB processed
        ("drug1_db", "drug_name", "drug_desc"),
        ("drug2_db", "disease_name", "disease_desc"),

        ("drugbank_id", "drug_name", "drug_desc"),
        ("doid_id", "disease_name", "disease_desc"),
    ]

    for db_col, name_col, desc_col in candidate_sets:
        if db_col not in df.columns:
            continue

        for _, row in df.iterrows():
            db = row.get(db_col, "")
            name = row.get(name_col, "") if name_col in df.columns else ""
            desc = row.get(desc_col, "") if desc_col in df.columns else ""

            add_drug_text(
                drug_texts=drug_texts,
                drugbank_id=db,
                name=name,
                desc=desc,
                max_chars=max_chars,
                text_mode=text_mode,
            )


def collect_entity_texts(
    paths: Dict[str, str],
    retrieval_json: str,
    max_chars: int,
    text_mode: str,
) -> Dict[str, str]:
    """
    Build DrugBank ID -> readable drug text.

    Priority:
        1. retrieval_json, e.g. d3.json
        2. train/test files
        3. id_to_name_mapping.csv
        4. fallback later to DrugBank ID
    """
    drug_texts: Dict[str, str] = {}

    if retrieval_json and os.path.exists(retrieval_json):
        logger.info(f"Loading retrieval JSON for drug text: {retrieval_json}")
        retrieval_df = read_table_auto(retrieval_json)
        collect_drug_texts_from_df(
            df=retrieval_df,
            source_name="retrieval_json",
            drug_texts=drug_texts,
            max_chars=max_chars,
            text_mode=text_mode,
        )
    else:
        raise FileNotFoundError(f"retrieval_json not found: {retrieval_json}")

    for split_name in ["train_set", "test_set"]:
        path = paths.get(split_name)

        if path and os.path.exists(path):
            logger.info(f"Loading {split_name}: {path}")
            df = read_table_auto(path)
            collect_drug_texts_from_df(
                df=df,
                source_name=split_name,
                drug_texts=drug_texts,
                max_chars=max_chars,
                text_mode=text_mode,
            )
        else:
            logger.warning(f"{split_name} not found: {path}")

    drug_info_path = paths.get("drug_info")

    if drug_info_path and os.path.exists(drug_info_path):
        logger.info(f"Loading entity name fallback: {drug_info_path}")

        if drug_info_path.endswith(".csv"):
            info = pd.read_csv(drug_info_path)

            if "drugbank_id" in info.columns and "name" in info.columns:
                for _, row in info.iterrows():
                    entity_id = safe_str(row.get("drugbank_id"))
                    name = safe_str(row.get("name"))

                    if entity_id and name and entity_id not in drug_texts:
                        drug_texts[entity_id] = name
            else:
                logger.warning(f"Could not infer CSV info columns: {list(info.columns)}")

        elif drug_info_path.endswith(".json"):
            info = load_json(drug_info_path)

            if isinstance(info, dict):
                for entity_id, value in info.items():
                    entity_id = safe_str(entity_id)

                    if isinstance(value, str):
                        text = clean_text(value)
                    elif isinstance(value, dict):
                        name = safe_str(value.get("name", ""))
                        desc = safe_str(value.get("description", ""))
                        text = f"{name}. {desc}".strip() if desc else name
                    else:
                        text = ""

                    if entity_id and text and entity_id not in drug_texts:
                        drug_texts[entity_id] = truncate_text(text, max_chars=max_chars)
            else:
                logger.warning(f"Unsupported JSON info structure in {drug_info_path}")

    logger.info(f"Collected entity text entries: {len(drug_texts)}")


    for db in [
        "DB12612",  # Ozanimod
        "DB12371",  # Siponimod
        "DB09053",  # Ibrutinib
        "DB00068",  # Interferon beta-1b
        "DB01109",  # Heparin
        "DB05889",  # Inotuzumab ozogamicin
        "DB11595",  # Atezolizumab
    ]:
        preview = drug_texts.get(db)
        logger.info(f"Preview {db}: {preview[:300] if preview else 'NOT FOUND'}")

    return drug_texts


def build_id_to_text(
    dataset_name: str,
    paths: Dict[str, str],
    retrieval_json: str,
    max_chars: int,
    text_mode: str,
) -> Dict[int, str]:
    """
    Build final node_id -> text mapping.
    """
    # if dataset_name != "ddinter":
    #     raise ValueError("This script currently supports only DDInter.")

    bkg_entity2id = load_json(paths["bkg_entity2id"])
    node2id = load_json(paths["node2id"])
    kg_nodes, _ = load_raw_kg_nodes(paths["augmented_kg"])

    hetionet_key_to_text = build_hetionet_key_to_text(paths["hetionet"])
    entity_texts = collect_entity_texts(
        paths=paths,
        retrieval_json=retrieval_json,
        max_chars=max_chars,
        text_mode=text_mode,
    )

    id_to_text: Dict[int, str] = {}

    # 1. BKG
    bkg_resolved = 0
    bkg_fallback = 0

    for key, nid in bkg_entity2id.items():
        nid = int(nid)
        key = safe_str(key)

        if key in hetionet_key_to_text:
            id_to_text[nid] = clean_text(hetionet_key_to_text[key])
            bkg_resolved += 1
        else:
            id_to_text[nid] = clean_text(key)
            bkg_fallback += 1

    logger.info(f"BKG nodes resolved using Hetionet names: {bkg_resolved}")
    logger.info(f"BKG nodes fallback to raw key: {bkg_fallback}")

    # 2. drug nodes
    drug_resolved = 0
    drug_fallback = 0

    for drugbank_id, nid in node2id.items():
        nid = int(nid)
        db = safe_str(drugbank_id)

        if db in entity_texts:
            id_to_text[nid] = clean_text(entity_texts[db])
            drug_resolved += 1
        else:
            id_to_text[nid] = clean_text(db)
            drug_fallback += 1

    logger.info(f"Dataset nodes resolved using name/description: {drug_resolved}")
    logger.info(f"Dataset nodes fallback to original ID: {drug_fallback}")

    # 3. Raw KG nodes missing from both mappings
    kg_fallback = 0

    for nid in kg_nodes:
        nid = int(nid)

        if nid not in id_to_text:
            id_to_text[nid] = f"Node {nid}"
            kg_fallback += 1

    logger.info(f"Raw KG nodes fallback to generic text: {kg_fallback}")

    logger.info(f"Total node texts: {len(id_to_text)}")
    logger.info(f"Raw KG node count: {len(kg_nodes)}")
    logger.info(f"BKG_entity2Id count: {len(bkg_entity2id)}")
    logger.info(f"node2id drug count: {len(node2id)}")

    node2id_int = {safe_str(k): int(v) for k, v in node2id.items()}

    for db in [
        "DB12612",
        "DB12371",
        "DB09053",
        "DB00068",
        "DB01109",
        "DB05889",
        "DB11595",
    ]:
        nid = node2id_int.get(db)

        if nid is not None:
            logger.info(
                f"Embedding text preview for {db} node_id={nid}: "
                f"{id_to_text.get(nid, '')[:400]}"
            )

    return id_to_text


def validate_embedding_coverage(
    node_embeddings: Dict[int, np.ndarray],
    paths: Dict[str, str],
    retrieval_json: str,
):
    """
    Validate that embeddings cover:
        - all raw KG nodes
        - all node2id drug nodes
        - cold-start missing endpoint nodes from retrieval_json
    """
    embedded_nodes = set(int(k) for k in node_embeddings.keys())

    kg_nodes, _ = load_raw_kg_nodes(paths["augmented_kg"])
    node2id = load_json(paths["node2id"])
    node2id_nodes = set(int(v) for v in node2id.values())

    logger.info("=" * 100)
    logger.info("Embedding coverage validation")
    logger.info("=" * 100)

    logger.info(f"Embedded nodes: {len(embedded_nodes)}")
    logger.info(f"Raw KG nodes: {len(kg_nodes)}")
    logger.info(f"node2id drug nodes: {len(node2id_nodes)}")

    logger.info(f"Raw KG nodes missing embeddings: {len(kg_nodes - embedded_nodes)}")
    logger.info(f"node2id drug nodes missing embeddings: {len(node2id_nodes - embedded_nodes)}")
    logger.info(f"Embedded nodes absent from raw KG: {len(embedded_nodes - kg_nodes)}")

    if kg_nodes - embedded_nodes:
        logger.warning(
            f"First 20 raw KG nodes missing embeddings: "
            f"{sorted(list(kg_nodes - embedded_nodes))[:20]}"
        )

    if node2id_nodes - embedded_nodes:
        logger.warning(
            f"First 20 node2id nodes missing embeddings: "
            f"{sorted(list(node2id_nodes - embedded_nodes))[:20]}"
        )

    df = read_table_auto(retrieval_json)

    if "coverage_diag" not in df.columns:
        logger.warning("retrieval_json has no coverage_diag; skipping cold-start validation.")
        return

    diag = pd.json_normalize(df["coverage_diag"])

    if "d1_in_graph" not in diag.columns or "d2_in_graph" not in diag.columns:
        logger.warning("coverage_diag missing d1_in_graph/d2_in_graph.")
        return

    tmp = df.copy()
    tmp["d1_in_graph"] = diag["d1_in_graph"].apply(safe_bool).values
    tmp["d2_in_graph"] = diag["d2_in_graph"].apply(safe_bool).values

    missing_ids = set()

    if "drug1_id" in tmp.columns:
        missing_ids |= set(tmp.loc[~tmp["d1_in_graph"], "drug1_id"].astype(int))

    if "drug2_id" in tmp.columns:
        missing_ids |= set(tmp.loc[~tmp["d2_in_graph"], "drug2_id"].astype(int))

    logger.info(f"Cold-start/missing endpoint IDs from retrieval_json: {len(missing_ids)}")
    logger.info(f"Cold-start/missing endpoint IDs in embeddings: {len(missing_ids & embedded_nodes)}")
    logger.info(f"Cold-start/missing endpoint IDs missing embeddings: {len(missing_ids - embedded_nodes)}")
    logger.info(f"Any -1 among missing endpoint IDs: {-1 in missing_ids}")

    if missing_ids - embedded_nodes:
        logger.warning(
            f"First 20 missing endpoint IDs without embeddings: "
            f"{sorted(list(missing_ids - embedded_nodes))[:20]}"
        )


def save_text_debug(id_to_text: Dict[int, str], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in id_to_text.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )


def main(args):
    paths = PATHS[args.dataset_name]

    id_to_text = build_id_to_text(
        dataset_name=args.dataset_name,
        paths=paths,
        retrieval_json=args.retrieval_json,
        max_chars=args.max_chars,
        text_mode=args.text_mode,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    text_debug_path = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_node_embedding_text_debug_{args.text_mode}.json",
    )
    save_text_debug(id_to_text, text_debug_path)
    logger.info(f"Saved text debug file to: {text_debug_path}")

    node_ids = []
    texts = []

    for nid, text in sorted(id_to_text.items()):
        node_ids.append(int(nid))
        texts.append(truncate_text(text, max_chars=args.max_chars))

    logger.info(f"Total nodes to embed: {len(node_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model: {args.model_name} on {device}")

    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = args.max_seq_length

    logger.info(
        f"Encoding with batch_size={args.batch_size}, "
        f"max_seq_length={args.max_seq_length}"
    )

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    node_embeddings = {
        int(nid): emb.astype(np.float32)
        for nid, emb in zip(node_ids, embeddings)
    }

    validate_embedding_coverage(
        node_embeddings=node_embeddings,
        paths=paths,
        retrieval_json=args.retrieval_json,
    )

    output_path = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_node_embeddings_{args.text_mode}.pkl",
    )

    with open(output_path, "wb") as f:
        pickle.dump(node_embeddings, f)

    logger.info(f"Saved embeddings to: {output_path}")
    logger.info(f"Total embeddings saved: {len(node_embeddings)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["ddinter", "pharmaDB", "drugbank"],
    )

    parser.add_argument(
        "--retrieval_json",
        type=str,
        required=True,
        help="Path to retrieval JSON, e.g. /kaggle/input/.../prompt/d3.json",
    )

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--max_chars", type=int, default=1500)
    parser.add_argument("--output_dir", type=str, default="/workspace/SemDrug/embeddings")

    parser.add_argument(
    "--text_mode",
    type=str,
    default="full",
    choices=["full", "name_only", "short"],
    help="How to build text for dataset-specific nodes.",
)

    args = parser.parse_args()
    main(args)