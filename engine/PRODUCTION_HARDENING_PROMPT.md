# Role: Staff Developer Tools Engineer
# Task: Harden GraphRAG Engine for Enterprise Scale (1M+ Lines)

## Context

I have a prototype "Static Artifact" engine (Indexer + MCP Server) that works on synthetic data.
I am deploying this to a **12-year-old iOS Monorepo** (Swift/Obj-C, 1M+ lines) on a **Mac M3 Max**.

### Architecture Philosophy: Map vs. Territory

- **Territory:** The actual file system (`.swift`, `.m`, `.h` files). This is the **only source of truth** for code content.
- **Map:** A lightweight SQLite index containing:
  - **Pointers:** File paths and byte-ranges (`start_byte`, `end_byte`)
  - **Topology:** Module relationships (Imports, Inheritance, Calls, Extensions)
  - **Vectors:** Embeddings of *signatures only* (not function bodies)

> [!IMPORTANT]
> We reject the "Live Database" approach (loading full code into VectorDB) because it creates **Cache Drift** — where the AI suggests code that was deleted 5 minutes ago.

---

## Goal

Refactor `indexer.py` and `server.py` to handle enterprise scale, dirty data, and incremental updates.

---

## Request 1: Harden `indexer.py` (The Harvester)

### 1.1 Incremental Indexing ("The Dirty Bit")

Create a SQLite table for change detection:

```sql
CREATE TABLE file_hashes (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Logic:**
1. On run, scan all allowed files and calculate SHA-256.
2. **Skip:** If `current_hash == stored_hash`, do not re-parse.
3. **Parse:** If hash differs or path is new, parse and update nodes/edges.
4. **Prune:** If a file exists in DB but not on disk, remove its nodes/edges.

> [!TIP]
> Parsing 1M lines from scratch takes too long. Incremental indexing is **mandatory**.

### 1.2 Rename Detection

Before treating a missing file as "deleted":
1. Check if any *new* file has the same hash as the orphaned DB entry.
2. If match found: Update the `path` in `file_hashes` — do not re-parse.
3. If no match: Treat as genuine deletion + new file creation.

### 1.3 Noise Filtration

Implement a strict `should_index(path: str) -> bool` function:

**Block (return False):**
- `Pods/`
- `Carthage/`
- `DerivedData/`
- `*.generated.swift`
- `*Tests/` (unit test directories)
- `*Tests.swift` (test files)
- `*Mock*.swift` (mock files)
- `.build/`
- `vendor/`

**Allow (return True):**
- `.swift`
- `.h`
- `.m`

### 1.4 Symlink & Path Handling

- Use `os.path.realpath()` to canonicalize all paths before hashing.
- Detect and skip symlink loops (max traversal depth: 10).
- Log warnings for broken symlinks to `indexing_errors.log`.
- Store **canonical paths** in the database, not relative or symlinked paths.

### 1.5 Parser Resilience

Wrap `tree-sitter` parsing in robust error handling:

```python
def parse_file(path: str) -> Optional[ParseResult]:
    try:
        # Attempt UTF-8 first
        content = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Legacy Obj-C files may use latin-1
        content = path.read_text(encoding='latin-1')
    
    try:
        tree = parser.parse(content.encode())
        return extract_symbols(tree)
    except Exception as e:
        log_error(path, e)
        return fallback_regex_parse(content)  # Graceful degradation
```

**Fallback Regex Parsing:**
If tree-sitter fails (e.g., `@Observable` macro, `#Preview`, Swift 6 syntax), extract symbols via regex:
- `func\s+(\w+)\s*\(`
- `class\s+(\w+)`
- `struct\s+(\w+)`
- `protocol\s+(\w+)`
- `extension\s+(\w+)`

Log all failures to `indexing_errors.log` with file path, line number, and error message. **Never crash the indexer.**

### 1.6 Parser Versioning

Pin tree-sitter-swift to a specific version in `pyproject.toml`:

```toml
[project.dependencies]
tree-sitter = ">=0.21.0"
tree-sitter-swift = "0.6.0"  # Pin specific version
```

Document known parse failures in a `PARSER_LIMITATIONS.md` file.

### 1.7 Transaction Safety

Wrap updates in SQLite transactions to prevent corruption on crash:

```python
def update_file_index(conn: sqlite3.Connection, file_path: str, symbols: List[Symbol]):
    with conn:  # Auto-commits on success, rolls back on exception
        conn.execute("DELETE FROM nodes WHERE file_path = ?", (file_path,))
        conn.execute("DELETE FROM edges WHERE source_file = ? OR target_file = ?", (file_path, file_path))
        conn.executemany("INSERT INTO nodes ...", symbols)
        conn.executemany("INSERT INTO edges ...", relationships)
        conn.execute("INSERT OR REPLACE INTO file_hashes (path, hash) VALUES (?, ?)", (file_path, new_hash))
```

For full re-index operations, use `BEGIN IMMEDIATE` to prevent readers from seeing partial state.

### 1.8 Parallelization

Use `ProcessPoolExecutor` to saturate M3 Max cores:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

MAX_WORKERS = os.cpu_count()  # M3 Max: 12 cores

def index_repository(repo_path: str):
    files = collect_indexable_files(repo_path)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(hash_and_parse, f): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            # Merge results into main thread's DB connection
```

> [!CAUTION]
> SQLite connections cannot be shared across processes. Each worker returns serializable results; the main thread writes to DB.

---

## Request 2: Implement the Weaver (Cross-File Linking)

After all files are parsed, run a **Weaver** phase on the main thread to resolve cross-file relationships.

### 2.1 Edge Types

Define explicit relationship types:

```sql
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    target_file TEXT,  -- NULL if unresolved
    target_symbol TEXT NOT NULL,
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'IMPORTS',    -- File A imports Module B
        'INHERITS',   -- Class A inherits from Class B
        'CONFORMS',   -- Type A conforms to Protocol B
        'CALLS',      -- Function A calls Function B (if extractable)
        'EXTENDS',    -- Extension A extends Type B
        'BRIDGING'    -- Swift class inherits Obj-C class
    )),
    line_number INTEGER,
    FOREIGN KEY (source_file) REFERENCES file_hashes(path)
);
```

### 2.2 Extension Resolution

Swift extensions scatter method definitions across files. Link them:

```sql
CREATE TABLE extension_map (
    extension_file TEXT NOT NULL,
    extension_symbol TEXT NOT NULL,  -- e.g., "User" from "extension User"
    canonical_file TEXT,             -- e.g., "User.swift" where "class User" is defined
    canonical_symbol TEXT,
    PRIMARY KEY (extension_file, extension_symbol)
);
```

**Logic:**
1. After parsing, collect all `extension Foo` declarations.
2. For each, find the canonical `class Foo` / `struct Foo` / `protocol Foo` definition.
3. Store the mapping for `trace_dependencies()` to surface as "scattered definitions."

### 2.3 Bridging Header Detection

Specifically track Swift-to-Obj-C inheritance:

```python
def detect_bridging(swift_class: Symbol, all_symbols: Dict[str, Symbol]) -> Optional[Edge]:
    if swift_class.inherits_from:
        parent = all_symbols.get(swift_class.inherits_from)
        if parent and parent.language == 'objc':
            return Edge(
                source=swift_class,
                target=parent,
                edge_type='BRIDGING'
            )
```

---

## Request 3: Embeddings (Neural Engine)

### 3.1 Model Selection

Use `sentence-transformers` with `nomic-embed-text-v1.5`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
model.to('mps')  # Use Metal Performance Shaders on M3 Max
```

**Why this model:**
- 768-dimensional vectors (good balance of quality vs. storage)
- Strong performance on code semantics
- Runs efficiently on Apple Silicon via MPS

### 3.2 Batching

Batch embeddings on the **main thread** (not in worker processes):

```python
BATCH_SIZE = 256

def embed_signatures(signatures: List[str]) -> List[np.ndarray]:
    embeddings = []
    for i in range(0, len(signatures), BATCH_SIZE):
        batch = signatures[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    return embeddings
```

### 3.3 Storage

Store embeddings as BLOBs in SQLite:

```sql
CREATE TABLE node_embeddings (
    node_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,  -- numpy .tobytes()
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);
```

```python
def store_embedding(conn: sqlite3.Connection, node_id: int, embedding: np.ndarray):
    conn.execute(
        "INSERT OR REPLACE INTO node_embeddings (node_id, embedding) VALUES (?, ?)",
        (node_id, embedding.astype(np.float32).tobytes())
    )
```

---

## Request 4: Enhance `server.py` (The MCP Interface)

### 4.1 Startup: Load Graph into RAM

On server start, load the topology into NetworkX for O(1) traversal:

```python
import networkx as nx

def load_graph(db_path: str) -> nx.DiGraph:
    G = nx.DiGraph()
    conn = sqlite3.connect(db_path)
    
    # Load nodes
    for row in conn.execute("SELECT id, file_path, symbol_name, symbol_type FROM nodes"):
        G.add_node(row[0], path=row[1], name=row[2], type=row[3])
    
    # Load edges
    for row in conn.execute("SELECT source_symbol, target_symbol, edge_type FROM edges"):
        G.add_edge(row[0], row[1], type=row[2])
    
    return G
```

### 4.2 Tool: `trace_dependencies(file_path)`

Find upstream (who depends on this?) and downstream (what does this depend on?) nodes:

```python
@mcp.tool()
def trace_dependencies(file_path: str) -> dict:
    """
    Trace all dependencies for a given file.
    Returns upstream (dependents) and downstream (dependencies) relationships.
    """
    nodes = [n for n, d in G.nodes(data=True) if d.get('path') == file_path]
    
    if not nodes:
        return {"error": f"File not found in index: {file_path}"}
    
    result = {
        "target": {
            "path": file_path,
            "symbols": [G.nodes[n]['name'] for n in nodes]
        },
        "upstream": [],    # Who depends on this file?
        "downstream": [],  # What does this file depend on?
        "extensions": []   # Scattered extension files
    }
    
    for node in nodes:
        # Upstream: predecessors in the graph
        for pred in G.predecessors(node):
            edge_data = G.edges[pred, node]
            result["upstream"].append({
                "path": G.nodes[pred].get('path'),
                "symbol": G.nodes[pred].get('name'),
                "edge_type": edge_data.get('type'),
            })
        
        # Downstream: successors in the graph
        for succ in G.successors(node):
            edge_data = G.edges[node, succ]
            result["downstream"].append({
                "path": G.nodes[succ].get('path'),
                "symbol": G.nodes[succ].get('name'),
                "edge_type": edge_data.get('type'),
            })
    
    # Extensions from extension_map
    extensions = conn.execute(
        "SELECT extension_file FROM extension_map WHERE canonical_file = ?",
        (file_path,)
    ).fetchall()
    result["extensions"] = [e[0] for e in extensions]
    
    return result
```

**Output Schema:**
```json
{
  "target": {
    "path": "/path/to/UserManager.swift",
    "symbols": ["UserManager", "UserManagerDelegate"]
  },
  "upstream": [
    {"path": "/path/to/AuthService.swift", "symbol": "AuthService", "edge_type": "CALLS"}
  ],
  "downstream": [
    {"path": "/path/to/User.swift", "symbol": "User", "edge_type": "IMPORTS"}
  ],
  "extensions": [
    "/path/to/UserManager+Networking.swift",
    "/path/to/UserManager+Persistence.swift"
  ]
}
```

### 4.3 Tool: `find_bridging_header_usage()`

Find Swift classes that inherit from Obj-C classes:

```python
@mcp.tool()
def find_bridging_header_usage() -> dict:
    """
    Find all Swift classes that inherit from Objective-C classes.
    Useful for understanding Swift/Obj-C interop boundaries.
    """
    bridging_edges = [
        (u, v, d) for u, v, d in G.edges(data=True) 
        if d.get('type') == 'BRIDGING'
    ]
    
    return {
        "count": len(bridging_edges),
        "bridging_classes": [
            {
                "swift_class": G.nodes[u].get('name'),
                "swift_file": G.nodes[u].get('path'),
                "objc_parent": G.nodes[v].get('name'),
                "objc_file": G.nodes[v].get('path'),
            }
            for u, v, d in bridging_edges
        ]
    }
```

### 4.4 Tool: `read_symbol(file_path, start_byte, end_byte)`

**The "Territory" retrieval** — read live code from disk:

```python
@mcp.tool()
def read_symbol(file_path: str, start_byte: int, end_byte: int) -> dict:
    """
    Read the actual source code for a symbol from disk.
    This ensures we never serve stale/cached code.
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(start_byte)
            content = f.read(end_byte - start_byte)
        
        # Attempt UTF-8, fallback to latin-1
        try:
            code = content.decode('utf-8')
        except UnicodeDecodeError:
            code = content.decode('latin-1')
        
        return {
            "file": file_path,
            "range": {"start": start_byte, "end": end_byte},
            "code": code
        }
    except FileNotFoundError:
        return {
            "error": "FILE_DELETED",
            "message": f"File no longer exists: {file_path}. Index may be stale."
        }
```

> [!IMPORTANT]
> This is the core of the "Map vs. Territory" pattern. The index tells us *where* to look; this tool reads the *actual current code*.

### 4.5 Tool: `semantic_search(query, top_k=10)`

Vector similarity search over signatures:

```python
@mcp.tool()
def semantic_search(query: str, top_k: int = 10) -> dict:
    """
    Find symbols semantically similar to the query.
    Searches over function/class signatures, not full code bodies.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Load all embeddings (consider FAISS for >100k nodes)
    results = []
    for row in conn.execute("SELECT n.id, n.file_path, n.symbol_name, e.embedding FROM nodes n JOIN node_embeddings e ON n.id = e.node_id"):
        node_id, path, name, emb_blob = row
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        results.append((similarity, node_id, path, name))
    
    results.sort(reverse=True)
    
    return {
        "query": query,
        "results": [
            {"score": float(sim), "file": path, "symbol": name}
            for sim, _, path, name in results[:top_k]
        ]
    }
```

---

## Request 5: Schema Definition

Create `schema.sql` with the complete database schema:

```sql
-- File change tracking for incremental indexing
CREATE TABLE file_hashes (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Symbol nodes (functions, classes, structs, protocols, extensions)
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL CHECK (symbol_type IN (
        'function', 'class', 'struct', 'protocol', 'extension', 
        'enum', 'property', 'initializer', 'typealias'
    )),
    language TEXT NOT NULL CHECK (language IN ('swift', 'objc')),
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    line_number INTEGER,
    signature TEXT,  -- The signature text for embedding
    FOREIGN KEY (file_path) REFERENCES file_hashes(path) ON DELETE CASCADE
);

-- Relationships between symbols
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER,  -- NULL if target is unresolved
    target_symbol TEXT NOT NULL,  -- For unresolved references
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'IMPORTS', 'INHERITS', 'CONFORMS', 'CALLS', 'EXTENDS', 'BRIDGING'
    )),
    line_number INTEGER,
    FOREIGN KEY (source_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES nodes(id) ON DELETE SET NULL
);

-- Extension-to-canonical-type mapping
CREATE TABLE extension_map (
    extension_node_id INTEGER NOT NULL,
    canonical_node_id INTEGER,
    FOREIGN KEY (extension_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (canonical_node_id) REFERENCES nodes(id) ON DELETE SET NULL,
    PRIMARY KEY (extension_node_id)
);

-- Vector embeddings for semantic search
CREATE TABLE node_embeddings (
    node_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_nodes_file ON nodes(file_path);
CREATE INDEX idx_nodes_symbol ON nodes(symbol_name);
CREATE INDEX idx_nodes_type ON nodes(symbol_type);
CREATE INDEX idx_edges_source ON edges(source_node_id);
CREATE INDEX idx_edges_target ON edges(target_node_id);
CREATE INDEX idx_edges_type ON edges(edge_type);
```

---

## Hardware Constraints (M3 Max)

| Component | Constraint |
|-----------|------------|
| **Hashing & Parsing** | `ProcessPoolExecutor`, `max_workers=cpu_count()` (12 cores) |
| **Embeddings** | Main thread only, batched (size 256), MPS device |
| **SQLite Writes** | Main thread only (workers return serializable results) |
| **RAM** | NetworkX graph loaded at server startup |

---

## Output Files

Provide the following production-ready files:

1. **`indexer_prod.py`** — The Harvester + Weaver implementation
2. **`server_prod.py`** — The MCP server with all tools
3. **`schema.sql`** — Complete database schema
4. **`pyproject.toml`** — Dependencies with pinned versions
5. **`PARSER_LIMITATIONS.md`** — Known tree-sitter-swift parse failures

---

## Success Criteria

- [ ] Full index of 1M+ line repo completes in < 30 minutes (first run)
- [ ] Incremental re-index (10 changed files) completes in < 10 seconds
- [ ] No crashes on malformed/legacy files
- [ ] `trace_dependencies()` returns in < 100ms
- [ ] `semantic_search()` returns in < 500ms
- [ ] Zero cache drift: `read_symbol()` always returns current disk content
