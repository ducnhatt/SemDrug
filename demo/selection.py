import os
import json
import shutil
from collections import defaultdict, Counter

# from IPython.display import display
import pandas as pd

BASE = "/kaggle/input/semdrug"
DEMO = "/kaggle/working/demo_semdrug"
DEMO_DATA = "demo/data"

os.makedirs(DEMO_DATA, exist_ok=True)

FULL_TEST = "data/pharmaDB/pharmaDB_test_set.json"
FULL_KG = "data/pharmaDB/pharmaDB_Augmented_KG.txt"
NODE2ID = "data/pharmaDB/node2id.json"

DEMO_TEST = f"{DEMO_DATA}/pharmaDB_test_set_demo.json"
DEMO_KG = f"{DEMO_DATA}/pharmaDB_Augmented_KG_demo.txt"


def read_json_auto(path):
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}. Existing columns: {list(df.columns)}")


# Load test set and node mapping
test_df = read_json_auto(FULL_TEST)

with open(NODE2ID, "r") as f:
    node2id = json.load(f)

drug_col = find_col(test_df, ["drugbank_id", "drug1_db", "Drug1_ID"])
disease_col = find_col(test_df, ["doid_id", "drug2_db", "Drug2_ID"])

test_df["drug1_id_demo"] = test_df[drug_col].map(node2id).fillna(-1).astype(int)
test_df["drug2_id_demo"] = test_df[disease_col].map(node2id).fillna(-1).astype(int)

test_df = test_df[
    (test_df["drug1_id_demo"] >= 0) &
    (test_df["drug2_id_demo"] >= 0)
].copy()

print("Usable test rows:", len(test_df))
print("Columns:", list(test_df.columns))


# Load full augmented KG
kg = pd.read_csv(
    FULL_KG,
    sep=r"\s+",
    names=["node1", "node2", "relation"],
    engine="python"
)

kg["node1"] = kg["node1"].astype(int)
kg["node2"] = kg["node2"].astype(int)
kg["relation"] = kg["relation"].astype(int)

# Compute endpoint degrees to pick examples that are likely to have graph context
deg = Counter()
for a, b in zip(kg["node1"].values, kg["node2"].values):
    deg[int(a)] += 1
    deg[int(b)] += 1

test_df["degsum_demo"] = (
    test_df["drug1_id_demo"].map(deg).fillna(0) +
    test_df["drug2_id_demo"].map(deg).fillna(0)
)

# Try to select a small balanced subset if label/category exists
label_col = None
for c in ["category", "label"]:
    if c in test_df.columns:
        label_col = c
        break

DEMO_N_PER_CLASS = 2

if label_col is not None:
    demo_df = (
        test_df.sort_values("degsum_demo", ascending=False)
        .groupby(label_col, group_keys=False)
        .head(DEMO_N_PER_CLASS)
    )
else:
    demo_df = test_df.sort_values("degsum_demo", ascending=False).head(6)

# Keep only a small number
demo_df = demo_df.head(6).copy()

# Save test subset without helper columns
drop_cols = ["drug1_id_demo", "drug2_id_demo", "degsum_demo"]
demo_save = demo_df.drop(columns=[c for c in drop_cols if c in demo_df.columns])

with open(DEMO_TEST, "w", encoding="utf-8") as f:
    json.dump(demo_save.to_dict("records"), f, ensure_ascii=False, indent=2)

print("Saved demo test:", DEMO_TEST)
print("Demo examples:")
print(demo_df[[drug_col, disease_col] + ([label_col] if label_col else [])].to_string())


## Build query-centered mini-KG by keeping neighborhoods around selected endpoints
# start_nodes = set(demo_df["drug1_id_demo"].astype(int)) | set(demo_df["drug2_id_demo"].astype(int))

# adj = defaultdict(list)
# for a, b in zip(kg["node1"].values, kg["node2"].values):
#     a = int(a)
#     b = int(b)
#     adj[a].append(b)
#     adj[b].append(a)

# def build_bfs_nodes(start_nodes, radius=2, cap_per_node=80):
#     visited = set(start_nodes)
#     frontier = list(start_nodes)

#     for _ in range(radius):
#         new_frontier = set()
#         for node in frontier:
#             neighbors = adj.get(node, [])
#             for nb in neighbors[:cap_per_node]:
#                 if nb not in visited:
#                     new_frontier.add(nb)
#         visited |= new_frontier
#         frontier = list(new_frontier)

#     return visited

# # Try a few caps to keep the demo KG small enough
# for cap in [80, 50, 30, 20]:
#     keep_nodes = build_bfs_nodes(start_nodes, radius=2, cap_per_node=cap)
#     mini_kg = kg[
#         kg["node1"].isin(keep_nodes) &
#         kg["node2"].isin(keep_nodes)
#     ].copy()

#     print(f"cap_per_node={cap}: nodes={len(keep_nodes)}, edges={len(mini_kg)}")

#     if len(mini_kg) <= 50000:
#         break

# mini_kg.to_csv(DEMO_KG, sep=" ", index=False, header=False)

# print("Saved demo KG:", DEMO_KG)
# print("Mini-KG edges:", len(mini_kg))
# print("Mini-KG nodes:", len(set(mini_kg.node1) | set(mini_kg.node2)))