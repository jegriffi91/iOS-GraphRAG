import os
import sqlite3
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict
from tqdm import tqdm

import tree_sitter_swift
import tree_sitter_objc
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
DB_PATH = "knowledge-graph.sqlite"
REPO_ROOT = "mock_repo"
MODEL_NAME = "all-MiniLM-L6-v2"

# --- WORKER: PARSING (CPU BOUND) ---
def parse_file(file_path: str) -> List[Dict]:
    """Reads file, extracts Symbols + Byte Ranges. Pure CPU task."""
    try:
        ext = Path(file_path).suffix
        if ext not in [".swift", ".h", ".m"]: return []
        
        with open(file_path, "rb") as f:
            code_bytes = f.read()
            
        # Initialize parser locally per process
        parser = Parser()
        if ext == ".swift":
            parser.language = Language(tree_sitter_swift.language())
            query_scm = """
            (class_declaration name: (type_identifier) @name) @def
            (protocol_declaration name: (type_identifier) @name) @def
            (extension_declaration type: (type_identifier) @name) @def
            (function_declaration name: (simple_identifier) @name) @def
            """
        else:
            parser.language = Language(tree_sitter_objc.language())
            query_scm = """
            (interface_declaration name: (identifier) @name) @def
            """
        
        tree = parser.parse(code_bytes)
        query = parser.language.query(query_scm)
        captures = query.captures(tree.root_node)
        
        nodes = []
        for node, tag in captures:
            if tag == "name":
                parent = node.parent
                # Capture the signature (first line)
                sig_end = code_bytes.find(b'\n', parent.start_byte)
                if sig_end == -1: sig_end = parent.end_byte
                sig_line = code_bytes[parent.start_byte:sig_end].decode('utf-8', errors='ignore')
                
                nodes.append({
                    "id": f"{file_path}::{node.text.decode('utf-8')}",
                    "name": node.text.decode('utf-8'),
                    "type": parent.type,
                    "file_path": file_path,
                    "start_byte": parent.start_byte,
                    "end_byte": parent.end_byte,
                    "signature": sig_line
                })
        return nodes
    except Exception as e:
        print(f"⚠️ Parse Error {file_path}: {e}")
        return []

# --- MAIN: WEAVING & EMBEDDING ---
def build_graph():
    start_t = time.time()
    
    # 1. Harvest (Parallel)
    files = [str(p) for p in Path(REPO_ROOT).rglob("*") if p.suffix in [".swift", ".h", ".m"]]
    print(f"🚜 Harvesting {len(files)} files...")
    
    all_nodes = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(parse_file, files), total=len(files)))
        for r in results:
            all_nodes.extend(r)
            
    # 2. Weave (Link Extensions to Bases)
    print(f"🕸️  Weaving {len(all_nodes)} nodes...")
    # Create a lookup for "Base Types"
    base_types = {n['name']: n['id'] for n in all_nodes if n['type'] in ['class_declaration', 'interface_declaration']}
    edges = []
    
    for node in all_nodes:
        # Heuristic: If this is 'extension AuthSession', link to 'AuthSession'
        if 'extension' in node['type']:
            target_id = base_types.get(node['name'])
            if target_id:
                edges.append((node['id'], target_id, "EXTENDS"))
                
        # (Future: Add dependency parsing here for CALLED_BY edges)

    # 3. Vectorize (Batch - GPU/MPS)
    print("🧠 Embedding Signatures (Batch)...")
    model = SentenceTransformer(MODEL_NAME)
    sigs = [n['signature'] for n in all_nodes]
    # M3 Max can handle large batches
    embeddings = model.encode(sigs, batch_size=128, show_progress_bar=True)

    # 4. Persist
    print("💾 Saving Map to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    with open("schema.sql") as f:
        conn.executescript(f.read())
        
    cursor = conn.cursor()
    
    # Bulk Insert Nodes
    node_data = []
    for i, node in enumerate(all_nodes):
        node_data.append((
            node['id'], node['name'], node['type'], node['file_path'], 
            node['start_byte'], node['end_byte'], node['signature'], 
            embeddings[i].tobytes()
        ))
    
    cursor.executemany("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?)", node_data)
    cursor.executemany("INSERT OR REPLACE INTO edges VALUES (?,?,?)", edges)
        
    conn.commit()
    conn.close()
    print(f"✅ Done in {time.time() - start_t:.2f}s")

if __name__ == "__main__":
    build_graph()