import json
import torch
import pickle
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# PATHS
NODE2ID_PATH = "data/pharmaDB/entity2id.json"
OUTPUT_PATH = "data/pharmaDB/node_embeddings.pkl"
MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

def main():
    print("Loading Node Map...")
    with open(NODE2ID_PATH, 'r') as f:
        node2id = json.load(f)
    
    # Invert to ID -> Name
    # node2id is { "Gene::123": 5, "Compound::ABC": 99 ... }
    # We need to sort by ID to ensure index alignment
    id2name = {v: k for k, v in node2id.items()}
    max_id = max(id2name.keys())
    
    print(f"Total nodes: {len(id2name)}")
    
    # Prepare text list. 
    # If an ID is missing (gaps), we use a placeholder.
    all_texts = []
    valid_indices = []
    
    for i in range(max_id + 1):
        if i in id2name:
            # Clean name: "Gene::1234" -> "Gene 1234" (simplistic)
            # Better: If you have a separate id_to_name mapping with real English names, USE IT HERE.
            # Assuming you might want to use the raw string for now:
            name = id2name[i] 
            all_texts.append(name)
            valid_indices.append(i)
        else:
            all_texts.append("unknown") # Placeholder

    print("Loading BERT...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    print("Encoding Nodes (this may take a while)...")
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)

    print(f"Saving embeddings (Size: {embeddings.nbytes / 1024 / 1024:.2f} MB)...")
    
    print("Saving to file...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
        
    print(f"Done. Saved shape: {embeddings.shape}")

if __name__ == "__main__":
    main()