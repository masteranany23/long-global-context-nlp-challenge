# index_build.py
import os
import pickle
import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

# ----------------------------
# Build index (already existing)
# ----------------------------
def build_index(children_table, openai_key=None, index_dir="./indexes"):
    os.makedirs(index_dir, exist_ok=True)

    # Use sentence-transformers locally (all-MiniLM-L6-v2)
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    # Add embeddings to the table
    embedded_table = children_table.with_columns(
        vector=embedder(pw.this.text)
    )
    
    # Save embedder info to disk
    embedder_info = {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": 384
    }
    with open(os.path.join(index_dir, "embedder_info.pkl"), "wb") as f:
        pickle.dump(embedder_info, f)
    
    print(f"Created index with {embedder_info['model_name']} embedder")
    
    # Return the embedded table with the embedder for later use
    return {"embedder": embedder, "table": embedded_table}


# ----------------------------
# Load existing indexes from disk
# ----------------------------
def load_indexes(index_dir="./indexes"):
    """
    Load the embedder information from disk.
    Note: In this implementation, we rebuild the index each time since
    Pathway works with in-memory streaming tables.
    """
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index directory {index_dir} does not exist.")

    embedder_info_path = os.path.join(index_dir, "embedder_info.pkl")
    if not os.path.exists(embedder_info_path):
        raise FileNotFoundError(f"Embedder info not found in {index_dir}")
    
    with open(embedder_info_path, "rb") as f:
        embedder_info = pickle.load(f)
    
    # Recreate the embedder
    embedder = SentenceTransformerEmbedder(model=embedder_info["model_name"])
    
    print(f"Loaded embedder: {embedder_info['model_name']}")
    
    # Return just the embedder - the table will be rebuilt
    return {"embedder": embedder, "embedder_info": embedder_info}
