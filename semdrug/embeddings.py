import os
import json
import pickle
import argparse
import logging

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PATHS = {
    "drugbank": {
        "bkg_entity2id": "/kaggle/input/kpaths/data/drugbank/BKG_entity2Id.json",
        "node2id": "/kaggle/input/kpaths/data/drugbank/node2id.json",
        "augmented_kg": "/kaggle/input/kpaths/data/drugbank/drugbank_Augmented_KG.txt",
    },
    "ddinter": {
        "bkg_entity2id": "/kaggle/input/kpaths/data/ddinter/BKG_entity2Id.json",
        "node2id": "/kaggle/input/kpaths/data/ddinter/node2id.json",
        "augmented_kg": "/kaggle/input/kpaths/data/ddinter/ddinter_Augmented_KG.txt",
    },
    "pharmaDB": {
        "bkg_entity2id": "/kaggle/input/kpaths/data/pharmaDB/BKG_entity2Id.json",
        "node2id": "/kaggle/input/kpaths/data/pharmaDB/node2id.json",
        "augmented_kg": "/kaggle/input/kpaths/data/pharmaDB/pharmaDB_Augmented_KG.txt",
    },
}

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(x):
    """
    Convert KG-style names into cleaner natural text.
    Example: Gene::ACHE -> Gene ACHE
    """
    return (
        str(x)
        .replace("::", " ")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def load_id_to_name(paths):
    """
    Merge BKG entity mapping and dataset-specific mapping into id -> name.
    Assumes both mappings are name -> id.
    """
    bkg_map = load_json(paths["bkg_entity2id"])
    dataset_map = load_json(paths["node2id"])

    logger.info(f"BKG nodes: {len(bkg_map)}")
    logger.info(f"Dataset nodes: {len(dataset_map)}")

    id_to_name = {}

    for name, nid in bkg_map.items():
        id_to_name[int(nid)] = name

    for name, nid in dataset_map.items():
        id_to_name[int(nid)] = name

    logger.info(f"Total unique node IDs with names: {len(id_to_name)}")
    return id_to_name


def validate_graph_coverage(augmented_kg_path, node_embeddings):
    """
    Check whether every node in the augmented KG has an embedding.
    """
    if not os.path.exists(augmented_kg_path):
        logger.warning(f"Augmented KG not found, skipping coverage check: {augmented_kg_path}")
        return

    kg = pd.read_csv(
        augmented_kg_path,
        sep=r"\s+",
        names=["node1", "node2", "relation"],
        engine="python",
    )

    graph_nodes = set(kg["node1"].astype(int)).union(set(kg["node2"].astype(int)))
    embedded_nodes = set(int(k) for k in node_embeddings.keys())

    missing = graph_nodes - embedded_nodes

    logger.info(f"Augmented KG nodes: {len(graph_nodes)}")
    logger.info(f"Embedded nodes: {len(embedded_nodes)}")
    logger.info(f"Missing graph-node embeddings: {len(missing)}")

    if missing:
        logger.warning(f"First 20 missing node IDs: {list(sorted(missing))[:20]}")


def main(args):
    logger.info(f"Building embeddings for dataset: {args.dataset_name}")

    paths = PATHS[args.dataset_name]

    id_to_name = load_id_to_name(paths)

    node_ids = []
    texts = []

    for nid, name in sorted(id_to_name.items()):
        node_ids.append(nid)
        texts.append(clean_text(name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model: {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    logger.info("Encoding nodes...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    node_embeddings = {
        int(nid): emb for nid, emb in zip(node_ids, embeddings)
    }

    validate_graph_coverage(paths["augmented_kg"], node_embeddings)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_node_embeddings.pkl",
    )

    with open(output_path, "wb") as f:
        pickle.dump(node_embeddings, f)

    logger.info(f"Saved embeddings to: {output_path}")
    logger.info(f"Total embeddings: {len(node_embeddings)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["drugbank", "ddinter", "pharmaDB"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/embeddings")

    args = parser.parse_args()
    main(args)