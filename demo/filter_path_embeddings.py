import os
import math
import logging
import argparse
import pickle
from functools import partial
from collections import Counter

import numpy as np
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer

import json

import re

from utils.Kpaths_utils import (
    load_and_process_mappings,
    load_and_process_dataset,
    build_graph_from_file,
    find_Kpaths,
    remove_leakage,
    build_relations_dict,
    find_Kpaths_no_filter,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PATHS = {
    "drugbank": {
    },
    "ddinter": {
        "inductive": "/kaggle/input/semdrug/data/ddinter/ddinter_Augmented_KG.txt",
        "bkg_entity2id": "/kaggle/input/semdrug/data/ddinter/BKG_entity2Id.json",
        "drug_info": "/kaggle/input/semdrug/data/ddinter/id_to_name_mapping.csv",
        "node2id": "/kaggle/input/semdrug/data/ddinter/node2id.json",
        "train_set": "/kaggle/input/semdrug/data/ddinter/ddinter_train_set.csv",
        "test_set": "/kaggle/input/semdrug/data/ddinter/ddinter_test_set.json",
        "embeddings": "/kaggle/input/semdrug/embeddings/ddinter_embeddings.pkl",
    },
    "pharmaDB": {
        "inductive": "/kaggle/input/semdrug/data/pharmaDB/pharmaDB_Augmented_KG.txt",
        "bkg_entity2id": "/kaggle/input/semdrug/data/pharmaDB/BKG_entity2Id.json",
        "drug_info": "/kaggle/input/semdrug/data/pharmaDB/id_to_name_mapping.json",
        "node2id": "/kaggle/input/semdrug/data/pharmaDB/node2id.json",
        "train_set": "/kaggle/input/semdrug/data/pharmaDB/pharmaDB_train_set.csv",
        "test_set": "/kaggle/input/semdrug/demo/data/pharmaDB_test_set_demo.json",
        "embeddings": "/kaggle/input/semdrug/embeddings/pharmaDB_node_embeddings.pkl",
    },
}
HETIO_NET_PATH = "/kaggle/input/semdrug/data/hetionet/hetionet-v1.0.json"
RELATIONS_PATH = "/kaggle/input/semdrug/data/relations_dicts_file_all.json"
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + 1e-12)


def norm_keys(d):
    return {int(k) if str(k).isdigit() else k: v for k, v in d.items()}

#helper to create relation id, --semantically related to-->
def get_new_relation_id(relation_id_to_name):
    """
    Create a relation ID that does not conflict with existing relation IDs.
    """
    int_keys = [
        int(k)
        for k in relation_id_to_name.keys()
        if str(k).isdigit()
    ]

    return max(int_keys) + 1 if int_keys else 10000


def get_edge_relation(G, u, v):
    """
    Robust relation getter for DiGraph or MultiDiGraph.
    Your current code assumes DiGraph, but this keeps the function safer.
    """
    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        return None

    if G.is_multigraph():
        # edge_data is usually {key: attr_dict}
        for _, attr in edge_data.items():
            if isinstance(attr, dict):
                return attr.get("relation")
        return None

    return edge_data.get("relation")

# ==================
# ==================
# add cold-start helper func
#this function only connects absent DDInter drugs to similar drugs that already exist in the KG.
def get_original_graph_nodes(G):
    """
    Snapshot original KG nodes before synthetic cold-start edges are added.
    """
    return set(G.nodes())


def get_graph_endpoint_nodes(original_graph_nodes, node2id):
    """
    Dataset endpoint nodes from node2id that already appear in the original KG.
    """
    return {
        int(nid)
        for _, nid in node2id.items()
        if int(nid) in original_graph_nodes
    }


def get_cold_start_endpoint_nodes(original_graph_nodes, node2id):
    """
    Dataset endpoint nodes from node2id that are absent from the original KG.
    These are true cold-start endpoint nodes.
    """
    return {
        int(nid)
        for _, nid in node2id.items()
        if int(nid) not in original_graph_nodes
    }


def add_cold_start_similarity_edges(
    G,
    node_embeddings,
    node2id,
    original_graph_nodes,
    similarity_relation_id,
    top_proxy=1,
    min_similarity=0.90,
):
    """
    Find and add proxy nodes with bidirectional edges
    """
    covered_nodes = sorted(get_graph_endpoint_nodes(original_graph_nodes, node2id))
    cold_nodes = sorted(get_cold_start_endpoint_nodes(original_graph_nodes, node2id))

    covered_nodes = [
        nid for nid in covered_nodes
        if nid in node_embeddings
    ]

    cold_nodes = [
        nid for nid in cold_nodes
        if nid in node_embeddings
    ]

    logger.info(f"KG-covered endpoint nodes with embeddings: {len(covered_nodes)}")
    logger.info(f"Cold-start endpoint nodes with embeddings: {len(cold_nodes)}")

    if not covered_nodes or not cold_nodes:
        logger.warning("No cold-start similarity edges added.")
        return []

    covered_matrix = np.stack([
        node_embeddings[nid]
        for nid in covered_nodes
    ])

    added_edges = []

    for cold_id in cold_nodes:
        cold_vec = node_embeddings[cold_id]
        sims = covered_matrix @ cold_vec

        order = np.argsort(-sims)

        added_for_cold = 0

        for idx in order:
            proxy_id = covered_nodes[idx]
            sim = float(sims[idx])

            if sim < min_similarity:
                break

            if proxy_id == cold_id:
                continue

            G.add_edge(cold_id, proxy_id, relation=similarity_relation_id)
            G.add_edge(proxy_id, cold_id, relation=similarity_relation_id)

            added_edges.append(
                {
                    "cold_id": int(cold_id),
                    "proxy_id": int(proxy_id),
                    "similarity": sim,
                }
            )

            added_for_cold += 1

            if added_for_cold >= top_proxy:
                break

    logger.info(f"Cold-start nodes connected: {len(set(e['cold_id'] for e in added_edges))}")
    logger.info(f"Synthetic directed edges added: {len(added_edges) * 2}")

    return added_edges


def remove_query_edges_temporarily(G, d1, d2):
    removed = []

    for u, v in [(d1, d2), (d2, d1)]:
        if not G.has_edge(u, v):
            continue

        if G.is_multigraph():
            edge_data = G.get_edge_data(u, v)
            for key, attr in list(edge_data.items()):
                removed.append((u, v, key, attr.copy()))
                G.remove_edge(u, v, key=key)
        else:
            attr = G.get_edge_data(u, v).copy()
            removed.append((u, v, None, attr))
            G.remove_edge(u, v)

    return removed


def restore_removed_edges(G, removed):
    for u, v, key, attr in removed:
        if G.is_multigraph():
            G.add_edge(u, v, key=key, **attr)
        else:
            G.add_edge(u, v, **attr)


def is_complete_path(path, start_node, end_node):
    return bool(path) and path[0][0] == start_node and path[-1][2] == end_node


def filter_invalid_or_leaky_paths(paths, start_node, end_node, min_hops=2, max_hops=3):
    """
    Tuple-level safety filter.
    Since query edges are removed before retrieval, this should rarely remove anything,
    but it protects against candidate generators that return stale/direct paths.
    """
    clean = []
    seen_node_seqs = set()

    for p in paths:
        if not p:
            continue

        if len(p) < min_hops or len(p) > max_hops:
            continue

        if not is_complete_path(p, start_node, end_node):
            continue

        # Remove direct edge
        if len(p) == 1 and p[0][0] == start_node and p[0][2] == end_node:
            continue

        node_seq = tuple([p[0][0]] + [step[2] for step in p])

        # contain repeated nodes
        if len(node_seq) != len(set(node_seq)):
            continue
        
        #duplicate node seq
        if node_seq in seen_node_seqs:
            continue

        seen_node_seqs.add(node_seq)
        clean.append(p)

    return clean


# ---------------------------------------------------------------------
# Relation informativeness
# ---------------------------------------------------------------------
def compute_relation_idf(G):
    """
    Common relation types are less informative.
    Rare relation types receive a higher IDF-like score.
    """
    rel_counts = Counter()

    if G.is_multigraph():
        for _, _, _, data in G.edges(keys=True, data=True):
            rel_counts[data.get("relation")] += 1
    else:
        for _, _, data in G.edges(data=True):
            rel_counts[data.get("relation")] += 1

    total_edges = max(1, G.number_of_edges())

    return {
        rel: math.log((total_edges + 1.0) / (count + 1.0))
        for rel, count in rel_counts.items()
    }

#--------------
#cand generation module

# nx return (x,y,z,t) --> (x,r1,y), (y,r2,z),...
def node_path_to_edge_path(G, node_path):
    tuples = []

    for u, v in zip(node_path[:-1], node_path[1:]):
        if not G.has_edge(u, v):
            return None

        relation_id = get_edge_relation(G, u, v)
        tuples.append((u, relation_id, v))

    return tuples


def enumerate_candidate_paths(
    G,
    start_node,
    end_node,
    max_length=3,
    candidate_pool=100,
    min_hops=2,
):
    """
    Fallback raw candidate generator using NetworkX shortest_simple_paths.
    This is only used if find_Kpaths_no_filter fails or returns no paths.
    """
    if not G.has_node(start_node) or not G.has_node(end_node):
        return []

    try:
        if not nx.has_path(G, start_node, end_node):
            return []
    except Exception:
        return []

    candidates = []

    try:
        path_generator = nx.shortest_simple_paths(G, start_node, end_node)
    except Exception:
        return []

    for node_path in path_generator:
        hops = len(node_path) - 1

        if hops > max_length:
            break

        if hops < min_hops:
            continue

        edge_path = node_path_to_edge_path(G, node_path)

        if edge_path is not None:
            candidates.append(edge_path)

        if len(candidates) >= candidate_pool:
            break

    return candidates


def get_unfiltered_candidate_paths(
    G,
    start_node,
    end_node,
    max_length=3,
    candidate_pool=100,
    min_hops=2,
):

    candidates = []

    try:
        candidates = find_Kpaths_no_filter(
            G,
            start_node,
            end_node,
            max_length=max_length,
            max_paths=candidate_pool,
        )
    except TypeError:
        try:
            candidates = find_Kpaths_no_filter(G, start_node, end_node)
        except Exception:
            candidates = []
    except Exception:
        candidates = []

    if not candidates:
        candidates = enumerate_candidate_paths(
            G,
            start_node,
            end_node,
            max_length=max_length,
            candidate_pool=candidate_pool,
            min_hops=min_hops,
        )

    return filter_invalid_or_leaky_paths(
        candidates,
        start_node,
        end_node,
        min_hops=min_hops,
        max_hops=max_length,
    )


#-------
# semantic neighbor

def get_top_semantic_successors(G, node_id, node_embeddings, q_vec, top_k=20):
    if not G.has_node(node_id):
        return []

    candidates = []

    for v in G.neighbors(node_id):
        if not G.has_edge(node_id, v):
            continue

        emb = node_embeddings.get(v)
        if emb is None:
            continue

        relation_id = get_edge_relation(G, node_id, v)
        score = float(np.dot(q_vec, emb))
        candidates.append((v, score, relation_id))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def get_top_semantic_predecessors(G, node_id, node_embeddings, q_vec, top_k=20):
    if not G.has_node(node_id):
        return []

    candidates = []

    for u in G.predecessors(node_id):
        if not G.has_edge(u, node_id):
            continue

        emb = node_embeddings.get(u)
        if emb is None:
            continue

        relation_id = get_edge_relation(G, u, node_id)
        score = float(np.dot(q_vec, emb))
        candidates.append((u, score, relation_id))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def get_semantic_neighbors(G, node_id, node_embeddings, q_vec, top_k=5):
    """
    Weak final fallback: one-hop semantic neighbors.
    These are partial evidence, not complete connecting paths.
    """
    if not G.has_node(node_id):
        return []

    candidates = []

    for v in G.neighbors(node_id):
        if not G.has_edge(node_id, v):
            continue

        emb = node_embeddings.get(v)
        if emb is None:
            continue

        relation_id = get_edge_relation(G, node_id, v)
        score = float(np.dot(q_vec, emb))
        candidates.append((v, score, relation_id))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [[(node_id, rel, v)] for v, score, rel in candidates[:top_k]]


# ------
# Path scoring

def path_semantic_score(
    path_tuples,
    node_embeddings,
    q_vec,
    G,
    relation_idf,
    alpha_mean=0.70,
    alpha_bridge=0.20,
    alpha_specificity=0.10,
    alpha_relation=0.05,
    length_penalty=0.03,
):

    if not path_tuples:
        return -1e9

    # Exclude final endpoint
    intermediate_nodes = [v for (_, _, v) in path_tuples[:-1]]

    sims = []
    specificities = []

    for n in intermediate_nodes:
        emb = node_embeddings.get(n)
        sim = float(np.dot(q_vec, emb)) if emb is not None else 0.0
        sims.append(sim)

        try:
            deg = G.degree(n)
            specificity = 1.0 / (1.0 + math.log1p(deg))
        except Exception:
            specificity = 0.0

        specificities.append(specificity)

    if sims:
        mean_sim = float(np.mean(sims))
        bridge_sim = float((sims[0] + sims[-1]) / 2.0)
    else:
        mean_sim = 0.0
        bridge_sim = 0.0

    specificity_bonus = float(np.mean(specificities)) if specificities else 0.0

    rel_scores = [
        relation_idf.get(r, 0.0)
        for _, r, _ in path_tuples
    ]
    relation_score = float(np.mean(rel_scores)) if rel_scores else 0.0

    # Normalize relation score 
    relation_score = relation_score / (1.0 + relation_score)

    score = (
        alpha_mean * mean_sim
        + alpha_bridge * bridge_sim
        + alpha_specificity * specificity_bonus
        + alpha_relation * relation_score
        - length_penalty * max(0, len(path_tuples) - 2)
    )

    return float(score)


# ------------
# MMR diversity
def relation_sequence(path):
    return tuple(step[1] for step in path)


def node_sequence(path):
    if not path:
        return tuple()
    return tuple([path[0][0]] + [step[2] for step in path])


def intermediate_node_set(path):
    if len(path) <= 1:
        return set()
    return set(step[2] for step in path[:-1])


def path_similarity(path_a, path_b):
    """
    Soft redundancy measure for MMR.

    We do not hard-remove paths with the same relation sequence, because
    different genes/proteins with the same relation pattern may still be
    biologically complementary.
    """
    rel_a = set(relation_sequence(path_a))
    rel_b = set(relation_sequence(path_b))

    node_a = intermediate_node_set(path_a)
    node_b = intermediate_node_set(path_b)

    rel_jaccard = len(rel_a & rel_b) / max(1, len(rel_a | rel_b))
    node_jaccard = len(node_a & node_b) / max(1, len(node_a | node_b))

    exact_rel_bonus = 1.0 if relation_sequence(path_a) == relation_sequence(path_b) else 0.0

# 0.45, 0.35, 0.2 gives ~68.25 with 0.35 mmr
# 0.35, 0.45, 0.1 gives ~70.63 with 0.2 mmr, 32 pool,
# now, 0.25 0.55 0.05
#try 0.3 0.55 0.1

    return (
        0.35 * rel_jaccard
        + 0.45 * node_jaccard
        + 0.1 * exact_rel_bonus
    )


def mmr_select_items(scored_items, max_final_paths=10, mmr_lambda=0.5, stop_delta=None, min_final_paths=6):
    selected = []
    remaining = scored_items[:]
    seen_node_seqs = set()

    best_global_score = scored_items[0]["score"] if scored_items else None

    while remaining and len(selected) < max_final_paths:
        best_item = None
        best_value = -1e18

        for item in remaining:
            if item["node_seq"] in seen_node_seqs:
                continue

            if not selected:
                value = item["score"]
            else:
                redundancy = max(
                    path_similarity(item["path"], selected_item["path"])
                    for selected_item in selected
                )
                value = item["score"] - mmr_lambda * redundancy

            if value > best_value:
                best_value = value
                best_item = item

        if best_item is None:
            break

        # optional stopping
        if (
            stop_delta is not None
            and len(selected) >= min_final_paths
            and best_global_score is not None
            and best_item["score"] < best_global_score - stop_delta
        ):
            break

        selected.append(best_item)
        seen_node_seqs.add(best_item["node_seq"])
        remaining.remove(best_item)

    return selected


#choose relevant non-redundant paths
def rerank_candidate_paths_mmr(
    candidate_paths,
    node_embeddings,
    q_vec,
    G,
    relation_idf,
    max_final_paths=10,
    score_threshold=0.0,
    disable_threshold_pruning=False,
    mmr_lambda=0.20,
    rank_penalty=0.0,
    score_floor_delta=None,
    min_mmr_candidates=20,
    sort_selected_by_score=True,
):
    scored_items = []

    for rank, path in enumerate(candidate_paths):
        #first, compute base semantic score for each cand path
        base_score = path_semantic_score(
            path,
            node_embeddings=node_embeddings,
            q_vec=q_vec,
            G=G,
            relation_idf=relation_idf,
        )

        score = base_score - rank_penalty * math.log1p(rank)

        if (not disable_threshold_pruning) and score < score_threshold:
            continue

        scored_items.append(
            {
                "score": score,
                "base_score": base_score,
                "rank": rank,
                "path": path,
                "rel_seq": relation_sequence(path),
                "node_seq": node_sequence(path),
            }
        )

    scored_items.sort(key=lambda x: x["score"], reverse=True)

    if not scored_items:
        return []

    # Optional quality floor
    if score_floor_delta is not None:
        top_score = scored_items[0]["score"]
        filtered = [
            item for item in scored_items
            if item["score"] >= top_score - score_floor_delta
        ]

        # Only apply the floor if enough candidates remain.
        if len(filtered) >= max(max_final_paths, min_mmr_candidates):
            scored_items = filtered
    #choose high-scoring paths, but penalize paths that are too similar to already selected ones
    selected_items = mmr_select_items(
        scored_items,
        max_final_paths=max_final_paths,
        mmr_lambda=mmr_lambda,
    )

    # score-ordered
    if sort_selected_by_score:
        selected_items.sort(key=lambda x: x["score"], reverse=True)

    return [item["path"] for item in selected_items]


#--------------------
# fallback
def semantic_bridge_fallback(
    G,
    start_node,
    end_node,
    node_embeddings,
    q_vec,
    relation_idf,
    max_final_paths=10,
    score_threshold=0.0,
    disable_threshold_pruning=False,
    top_k_side=20,
    mmr_lambda=0.35,
):
    """
    Strong fallback:
      start -> x -> end
      start -> x -> y -> end

    This still returns complete paths, unlike local semantic-neighbor fallback.
    """
    if not G.has_node(start_node) or not G.has_node(end_node):
        return []

    left = get_top_semantic_successors(
        G, start_node, node_embeddings, q_vec, top_k=top_k_side
    )
    right = get_top_semantic_predecessors(
        G, end_node, node_embeddings, q_vec, top_k=top_k_side
    )

    bridge_candidates = []
    seen_node_seqs = set()

    right_nodes = {node for node, _, _ in right}

    # 2-hop bridge: start -> x -> end
    for x, _, rel_start_to_x in left:
        if x in right_nodes and G.has_edge(x, end_node):
            rel_x_to_end = get_edge_relation(G, x, end_node)
            path = [
                (start_node, rel_start_to_x, x),
                (x, rel_x_to_end, end_node),
            ]

            ns = node_sequence(path)
            if ns not in seen_node_seqs:
                bridge_candidates.append(path)
                seen_node_seqs.add(ns)

    # 3-hop bridge: start -> x -> y -> end
    for x, _, rel_start_to_x in left:
        for y, _, rel_y_to_end in right:
            if x == y:
                continue

            if G.has_edge(x, y):
                rel_x_to_y = get_edge_relation(G, x, y)

                path = [
                    (start_node, rel_start_to_x, x),
                    (x, rel_x_to_y, y),
                    (y, rel_y_to_end, end_node),
                ]

                ns = node_sequence(path)
                if ns not in seen_node_seqs:
                    bridge_candidates.append(path)
                    seen_node_seqs.add(ns)

    if not bridge_candidates:
        return []

    bridge_candidates = filter_invalid_or_leaky_paths(
        bridge_candidates,
        start_node,
        end_node,
        min_hops=2,
        max_hops=3,
    )

    return rerank_candidate_paths_mmr(
        bridge_candidates,
        node_embeddings=node_embeddings,
        q_vec=q_vec,
        G=G,
        relation_idf=relation_idf,
        max_final_paths=max_final_paths,
        score_threshold=score_threshold,
        disable_threshold_pruning=disable_threshold_pruning,
        mmr_lambda=mmr_lambda,
    )

def get_entities_for_leakage(example, dataset_name):
    if dataset_name == "pharmaDB":
        return example.get("drug_name"), example.get("disease_name")
    return example.get("drug1_name"), example.get("drug2_name")


def count_formatted_paths(path_str):
    return sum(1 for line in path_str.splitlines() if line.strip())


# ---------------------------------------------------------------------
# Main
def main(args):
    logger.info(f"Starting Semantic-Diverse Path Retrieval for: {args.dataset_name}")

    node_id_to_name = load_and_process_mappings(
        args.dataset_name,
        PATHS,
        HETIO_NET_PATH,
    )

    relation_id_to_name, offset, dataset_relations = build_relations_dict(
        args.dataset_name,
        RELATIONS_PATH,
        args.add_reverse_edges,
    )
#=======================add new===============
    similarity_relation_id = None

    if args.add_cold_start_similarity_edges:
        if args.dataset_name != "ddinter":
            raise ValueError(
                "Cold-start similarity edges are currently intended only for DDInter."
            )

        similarity_relation_id = get_new_relation_id(relation_id_to_name)
        relation_id_to_name[similarity_relation_id] = "{u} is semantically similar to {v}"

        logger.info(
            f"Added synthetic similarity relation: "
            f"id={similarity_relation_id}, template='{relation_id_to_name[similarity_relation_id]}'"
        )
#=======================add new===============

    dataset = load_and_process_dataset(
        args.dataset_name,
        args.split,
        PATHS,
        args.debug,
    )

    G = build_graph_from_file(
        PATHS[args.dataset_name]["inductive"],
        dataset,
        args.add_reverse_edges,
        offset,
    )
    original_graph_nodes = get_original_graph_nodes(G)

    logger.info(
        f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    node_names = norm_keys(node_id_to_name)
    rel_names = norm_keys(relation_id_to_name)
    dataset_dict = {
        idx: val
        for idx, val in enumerate(norm_keys(dataset_relations).values())
    }

    #rare relations get small bonus
    # relation_idf = compute_relation_idf(G)

    embeddings_path = args.embeddings_path or PATHS[args.dataset_name]["embeddings"]

    logger.info(f"Loading node embeddings from: {embeddings_path}")

    with open(embeddings_path, "rb") as f:
        raw_embeddings = pickle.load(f)

    node_embeddings = {
        int(k) if str(k).isdigit() else k: l2_normalize(v)
        for k, v in raw_embeddings.items()
    }

    #=======================add new===============

    with open(PATHS[args.dataset_name]["node2id"], "r", encoding="utf-8") as f:
        node2id = json.load(f)

    cold_start_edges = []

    if args.add_cold_start_similarity_edges:
        cold_start_edges = add_cold_start_similarity_edges(
            G=G,
            node_embeddings=node_embeddings,
            node2id=node2id,
            original_graph_nodes=original_graph_nodes,
            similarity_relation_id=similarity_relation_id,
            top_proxy=args.cold_start_top_proxy,
            min_similarity=args.cold_start_min_similarity,
        )

        os.makedirs(args.output_dir, exist_ok=True)

        # edge_debug_path = os.path.join(
        #     args.output_dir,
        #     f"{args.dataset_name}_cold_start_similarity_edges_top{args.cold_start_top_proxy}_sim{args.cold_start_min_similarity}.json",
        # )

        # with open(edge_debug_path, "w", encoding="utf-8") as f:
        #     json.dump(cold_start_edges, f, indent=2)

        # logger.info(f"Saved cold-start edge debug file to: {edge_debug_path}")

    # Compute relation IDF after optional synthetic edges are added.
    relation_idf = compute_relation_idf(G)

    #=======================add new===============

    def short_text(x, max_chars=500):
        x = re.sub(r"\s+", " ", str(x or "")).strip()
        return x[:max_chars]

    def build_query_text(example, dataset_name):
        if dataset_name == "pharmaDB":
            return (
                f"{example.get('drug_name', '')}: {example.get('drug_desc', '')}\n"
                f"{example.get('disease_name', '')}: {example.get('disease_desc', '')}\n"
                f"Find biologically meaningful paths that represent the relation between "
                f"{example.get('drug_name', '')} and {example.get('disease_name', '')}."
            )
        elif dataset_name in ["ddinter", "drugbank"]:
            d1 = example.get("drug1_name", "")
            d2 = example.get("drug2_name", "")

            return (
                f"Task: retrieve drug-drug interaction severity evidence for {d1} and {d2}.\n"
                f"Goal: find complete KG paths useful for distinguishing Major, Moderate, and Minor interaction severity.\n"
                f"Prioritize pharmacokinetic and pharmacodynamic mechanisms: CYP metabolism, enzyme inhibition or induction, transporters, serum concentration changes, QT prolongation, arrhythmia, bleeding risk, anticoagulant or antiplatelet effects, CNS depression, respiratory depression, hepatotoxicity, nephrotoxicity, cardiotoxicity, hypotension, hyperkalemia, contraindication, shared targets, and strong pharmacologic similarity.\n"
                f"Downweight generic shared mild side effects unless they support severity.\n"
                f"Drug 1: {d1}. Description: {short_text(example.get('drug1_desc', ''))}\n"
                f"Drug 2: {d2}. Description: {short_text(example.get('drug2_desc', ''))}"
            )

        return ""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    query_texts = [build_query_text(ex, args.dataset_name) for ex in dataset]

    #embed all queries at once for efficiency, instead of per-example in the retrieval loop

    query_embeddings = model.encode(
        query_texts,
        batch_size=args.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    def get_clean_string(current_paths, example):
        if not current_paths:
            return ""

        temp_str = ""

        for p in current_paths[: args.max_final_paths]:
            line = ""

            for i, (u, r, v) in enumerate(p):
                u_n = node_names.get(u, f"N-{u}")
                v_n = node_names.get(v, f"N-{v}")
                r_n = rel_names.get(r, "related to")

                try:
                    fmt = r_n.format(u=u_n, v=v_n)
                except Exception:
                    fmt = f"{u_n} {r_n} {v_n}"

                line += fmt if i == 0 else f" and {fmt}"

            temp_str += line + "\n"

        d1_name, d2_name = get_entities_for_leakage(example, args.dataset_name)

        return remove_leakage(
            {
                "path_str": temp_str,
                "label": example["label_idx"],
                "d1": d1_name,
                "d2": d2_name,
            },
            dataset_dict,
            "label",
            "d1",
            "d2",
        )
    def safe_shortest_len(G, source, target, cutoff=4, directed=True):
        if not G.has_node(source) or not G.has_node(target):
            return None

        H = G if directed else G.to_undirected(as_view=True)

        try:
            lengths = nx.single_source_shortest_path_length(
                H,
                source,
                cutoff=cutoff,
            )
            return lengths.get(target, f">{cutoff}_or_no_path")
        except nx.NetworkXNoPath:
            return None
        except Exception:
            return "error"


    def wrapped_get_formatted_paths(example, idx, mode="semantic_mmr"):
        d1, d2 = example["drug1_id"], example["drug2_id"]
        q_vec = query_embeddings[idx]

        retrieval_source = "none"
        paths = []
        candidate_count = 0
        used_leakage_safety_recovery = False
        complete_evidence = False

        #remove direct query edges before retrieval
        removed_edges = remove_query_edges_temporarily(G, d1, d2)

        try:
            # ---------------------------------------------------------
            # Main retrieval
            # ---------------------------------------------------------
            coverage_diag = {
                "d1_in_graph": G.has_node(d1),
                "d2_in_graph": G.has_node(d2),
                "d1_degree": G.degree(d1) if G.has_node(d1) else 0,
                "d2_degree": G.degree(d2) if G.has_node(d2) else 0,
                "d1_in_original_kg": d1 in original_graph_nodes,
                "d2_in_original_kg": d2 in original_graph_nodes,
                "d1_was_cold_start": d1 not in original_graph_nodes,
                "d2_was_cold_start": d2 not in original_graph_nodes,
            }
            if mode in ["semantic_mmr", "heuristic"]:
                if args.disable_semantic_scoring:
                    # original k-paths logic
                    paths = find_Kpaths(
                        G,
                        d1,
                        d2,
                        max_length=args.max_depth,
                        max_paths=args.max_final_paths,
                    )
                    paths = filter_invalid_or_leaky_paths(
                        paths,
                        d1,
                        d2,
                        min_hops=args.min_hops,
                        max_hops=args.max_depth,
                    )

                    if paths:
                        retrieval_source = "kpaths_no_semantic"
                        complete_evidence = True

                else:
                    candidate_paths = get_unfiltered_candidate_paths(
                        G,
                        d1,
                        d2,
                        max_length=args.max_depth,
                        candidate_pool=args.candidate_pool,
                        min_hops=args.min_hops,
                    )
                    candidate_count = len(candidate_paths)

                    paths = rerank_candidate_paths_mmr(
                        candidate_paths,
                        node_embeddings=node_embeddings,
                        q_vec=q_vec,
                        G=G,
                        relation_idf=relation_idf,
                        max_final_paths=args.max_final_paths,
                        score_threshold=args.pruning_threshold,
                        disable_threshold_pruning=args.disable_threshold_pruning,
                        mmr_lambda=args.mmr_lambda,
                        rank_penalty=args.rank_penalty,
                        score_floor_delta=args.score_floor_delta,
                        sort_selected_by_score=not args.no_sort_selected_by_score,
                    )

                    if paths:
                        retrieval_source = "semantic_mmr_reranked_paths"
                        complete_evidence = True

            elif mode == "K-paths":
                paths = find_Kpaths(
                    G,
                    d1,
                    d2,
                    max_length=args.max_depth,
                    max_paths=args.max_final_paths,
                )
                paths = filter_invalid_or_leaky_paths(
                    paths,
                    d1,
                    d2,
                    min_hops=args.min_hops,
                    max_hops=args.max_depth,
                )

                if paths:
                    retrieval_source = "kpaths"
                    complete_evidence = True

            # ---------------------------------------------------------
            # fall back
            if not paths and not args.disable_fallback:
                paths = semantic_bridge_fallback(
                    G,
                    d1,
                    d2,
                    node_embeddings=node_embeddings,
                    q_vec=q_vec,
                    relation_idf=relation_idf,
                    max_final_paths=args.max_final_paths,
                    score_threshold=args.pruning_threshold,
                    disable_threshold_pruning=args.disable_threshold_pruning,
                    top_k_side=args.top_k_side,
                    mmr_lambda=args.mmr_lambda,
                )

                if paths:
                    retrieval_source = "semantic_bridge_fallback"
                    complete_evidence = True

            # ---------------------------------------------------------
            # Weak partial fallback
            # ---------------------------------------------------------
            if not paths and not args.disable_partial_fallback:
                per_side = max(1, args.max_final_paths // 2)

                paths = (
                    get_semantic_neighbors(
                        G,
                        d1,
                        node_embeddings,
                        q_vec,
                        top_k=per_side,
                    )
                    + get_semantic_neighbors(
                        G,
                        d2,
                        node_embeddings,
                        q_vec,
                        top_k=per_side,
                    )
                )

                if paths:
                    retrieval_source = "partial_semantic_neighbors"
                    complete_evidence = False

            # final path leakage using remove_leakage
            filtered_path_str = get_clean_string(paths, example)

            path_count = count_formatted_paths(filtered_path_str)

            if path_count == 0:
                complete_evidence = False

            if paths and not filtered_path_str.strip():
                used_leakage_safety_recovery = True
                logger.info(
                    f"String-level leakage safety triggered for "
                    f"{example.get('drug_name', example.get('drug1_name', d1))}"
                )

                # Try complete bridge recovery first.
                recovery_paths = semantic_bridge_fallback(
                    G,
                    d1,
                    d2,
                    node_embeddings=node_embeddings,
                    q_vec=q_vec,
                    relation_idf=relation_idf,
                    max_final_paths=args.max_final_paths,
                    score_threshold=args.pruning_threshold,
                    disable_threshold_pruning=args.disable_threshold_pruning,
                    top_k_side=args.top_k_side,
                    mmr_lambda=args.mmr_lambda,
                )

                recovery_str = get_clean_string(recovery_paths, example)

                if recovery_paths and recovery_str.strip():
                    paths = recovery_paths
                    filtered_path_str = recovery_str
                    retrieval_source = "semantic_bridge_after_leakage_safety"
                    complete_evidence = True

            result = {
                "all_paths": paths,
                "path_str": filtered_path_str,
                "raw_path_count": len(paths),
                "path_count": count_formatted_paths(filtered_path_str),
                "retrieval_source": retrieval_source,
                "complete_evidence": complete_evidence,
                "coverage_diag": coverage_diag,
                "candidate_count": candidate_count,
                "used_leakage_safety_recovery": used_leakage_safety_recovery,
            }

        finally:
            restore_removed_edges(G, removed_edges)

        return result

    wrapped_fn = partial(wrapped_get_formatted_paths, mode=args.mode)

    dataset = dataset.map(
        wrapped_fn,
        with_indices=True,
        num_proc=1,
    )

    suffix = []
    suffix.append(f"d{args.max_depth}")
    suffix.append(f"pool{args.candidate_pool}")
    suffix.append(f"k{args.max_final_paths}")
    suffix.append(f"mmr{args.mmr_lambda}")
    # suffix.append(f"pena{args.rank_penalty}")
    if args.score_floor_delta is not None:
        suffix.append(f"delta{args.score_floor_delta}")
    if args.stop_delta is not None:
        suffix.append(f"stop{args.stop_delta}")

    if args.disable_semantic_scoring:
        suffix.append("no_semscore")
    if args.disable_threshold_pruning:
        suffix.append("no_thresh")
    if args.disable_fallback:
        suffix.append("no_bridge_fallback")
    if args.disable_partial_fallback:
        suffix.append("no_partial_fallback")
    
    if args.add_cold_start_similarity_edges:
        suffix.append(
            f"coldtop{args.cold_start_top_proxy}sim{args.cold_start_min_similarity}"
        )

    suffix_str = "_" + "_".join(suffix)

    output_file = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_{args.mode}{suffix_str}.json",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.to_json(output_file)

    logger.info(f"Success! Processed data saved at {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pharmaDB",
        choices=["pharmaDB", "drugbank", "ddinter"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--add_reverse_edges", action="store_true")

    parser.add_argument(
        "--mode",
        type=str,
        default="semantic_mmr",
        choices=["semantic_mmr", "heuristic", "K-paths"],
    )

    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Optional override path for node embeddings pickle.",
    )

    # Retrieval settings
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_hops", type=int, default=2)
    parser.add_argument("--candidate_pool", "--beam_width", dest="candidate_pool", type=int, default=32)
    parser.add_argument("--max_final_paths", type=int, default=10)

    # Semantic filtering
    parser.add_argument("--pruning_threshold", type=float, default=0.0)
    parser.add_argument("--disable_threshold_pruning", action="store_true")
    parser.add_argument("--disable_semantic_scoring", action="store_true")

    # MMR
    parser.add_argument("--mmr_lambda", type=float, default=0.5)

    # Fallbacks
    parser.add_argument("--top_k_side", type=int, default=20)
    parser.add_argument("--disable_fallback", action="store_true")
    parser.add_argument("--disable_partial_fallback", action="store_true")

    # Runtime
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--rank_penalty", type=float, default=0.0)
    parser.add_argument("--score_floor_delta", type=float, default=None)
    parser.add_argument("--no_sort_selected_by_score", action="store_true")
    parser.add_argument("--stop_delta", type=float, default=None)

    parser.add_argument(
        "--add_cold_start_similarity_edges",
        action="store_true",
        help="Add synthetic similarity edges from true cold-start DDInter drugs to KG-covered proxy drugs.",
    )

    parser.add_argument(
        "--cold_start_top_proxy",
        type=int,
        default=1,
        help="Number of KG-covered proxy drugs to connect each cold-start drug to.",
    )

    parser.add_argument(
        "--cold_start_min_similarity",
        type=float,
        default=0.90,
        help="Minimum cosine similarity required to add a cold-start proxy edge.",
    )

    args = parser.parse_args()
    main(args)