import os
import ssl
import sqlite3
import networkx as nx
import numpy as np
from mcp.server.fastmcp import FastMCP

# Bypass SSL verification for enterprise proxies
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ['CURL_CA_BUNDLE'] = ''

mcp = FastMCP("iOS-Architect")

# --- IN-MEMORY STATE ---
GRAPH = nx.DiGraph()
NODE_MAP = {}   # ID -> Metadata
VECTORS = None  # Matrix [N, 384]
NODE_IDS = []   # List[ID] matching VECTORS rows

def hydrate():
    """Warm-up: Load SQLite Map into RAM."""
    print("🔥 Hydrating In-Memory Graph...")
    db_path = os.getenv("GRAPH_DB_PATH", "knowledge-graph.sqlite")
    conn = sqlite3.connect(db_path)
    
    # Load Nodes & Vectors
    # Explicitly naming columns to avoid "too many values to unpack" if schema has evolved
    query = "SELECT id, name, type, file_path, start_byte, end_byte, signature, vector FROM nodes"
    rows = conn.execute(query).fetchall()
    vec_list = []
    
    for row in rows:
        nid, name, ntype, path, start, end, sig, vec_blob = row
        
        GRAPH.add_node(nid, type=ntype, name=name)
        NODE_MAP[nid] = {"path": path, "range": (start, end), "sig": sig}
        
        vec = np.frombuffer(vec_blob, dtype=np.float32)
        vec_list.append(vec)
        NODE_IDS.append(nid)
        
    # Load Edges
    for src, tgt, rel in conn.execute("SELECT source, target, relation FROM edges"):
        GRAPH.add_edge(src, tgt, relation=rel)
        
    # Prepare Matrix
    global VECTORS
    if vec_list:
        VECTORS = np.array(vec_list)
        # Normalize for cosine similarity
        norm = np.linalg.norm(VECTORS, axis=1, keepdims=True)
        VECTORS = VECTORS / (norm + 1e-10)
        
    conn.close()
    print(f"⚡ Ready. {len(NODE_IDS)} nodes, {GRAPH.number_of_edges()} edges.")

# --- TOOLS ---

@mcp.tool()
def search_architecture(query: str) -> str:
    """Finds entry points using Vector Search."""
    # Note: In production, load the model globally. 
    # For script simplicity, we assume we can reload or use a lightweight encoder here
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    q_vec = model.encode([query])[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    
    # Cosine Similarity: O(1) Matrix mult
    scores = np.dot(VECTORS, q_vec)
    top_k_indices = np.argsort(scores)[-5:][::-1]
    
    results = []
    for idx in top_k_indices:
        nid = NODE_IDS[idx]
        results.append(f"[{scores[idx]:.2f}] {nid} ({NODE_MAP[nid]['sig']})")
        
    return "\n".join(results)

@mcp.tool()
def read_live_code(node_id: str) -> str:
    """
    Reads the LIVE implementation from disk. 
    Crucial: Reads the actual file, not a DB copy.
    """
    meta = NODE_MAP.get(node_id)
    if not meta: return "Node not found."
    
    try:
        with open(meta["path"], "r") as f:
            f.seek(meta["range"][0])
            length = meta["range"][1] - meta["range"][0]
            code = f.read(length)
            return f"File: {meta['path']}\nSnippet:\n```swift\n{code}\n```"
    except Exception as e:
        return f"Disk Read Error: {e}"

@mcp.tool()
def trace_impact(node_id: str) -> str:
    """Graph traversal to find dependencies."""
    if node_id not in GRAPH: return "Node not found."
    
    # Who depends on this? (Predecessors in a dependency graph, or Successors depending on edge direction)
    # If edge is (Child -> EXTENDS -> Parent), then Parent modifications impact Child.
    # So we look for Predecessors of Parent.
    
    impacted = list(nx.predecessors(GRAPH, node_id))
    return f"Node: {node_id}\nDirectly Impacted By Change: {impacted}"

if __name__ == "__main__":
    hydrate()
    mcp.run()
