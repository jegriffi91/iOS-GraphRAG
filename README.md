# iOS-GraphRAG

A semantic code intelligence engine for iOS/macOS codebases, built on the **"Map vs. Territory"** architecture. This MCP (Model Context Protocol) server provides AI-powered tools for navigating, searching, and understanding large Swift/Objective-C repositories (1M+ lines).

## Architecture Philosophy: Map vs. Territory

The engine is designed around a fundamental principle:

- **Territory:** The actual source files (`.swift`, `.m`, `.h`) on disk — the *only* source of truth for code content
- **Map:** A lightweight SQLite index containing:
  - **Pointers:** File paths and byte-ranges for surgical code retrieval
  - **Topology:** Relationships (inheritance, conformance, extensions, bridging)
  - **Vectors:** Embeddings of *signatures only* for semantic search

> [!IMPORTANT]
> This architecture deliberately avoids "Cache Drift" — where a live database serves code that was deleted or modified, causing AI hallucinations about non-existent implementations.

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (FastMCP)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │   NetworkX  │  │  Semantic   │  │   Territory Reader   │ │
│  │    Graph    │  │   Search    │  │   (Live Disk I/O)    │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  SQLite "Map" Database                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Nodes   │  │  Edges   │  │Extension │  │ Embeddings  │  │
│  │(Symbols) │  │(Relations)│  │   Map   │  │  (Vectors)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## MCP Tools

The server exposes four tools to AI clients via the Model Context Protocol:

### `semantic_search`
Find code by **meaning**, not just text matching. Uses AI embeddings for semantic similarity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `string` | Natural language description of what you're looking for |
| `top_k` | `int` | Number of results to return (default: 10) |

**USE THIS INSTEAD OF:** `grep -r "pattern"`, `find . -name "*.swift"`, or browsing files.

---

### `trace_dependencies`
Trace inheritance, protocol conformance, and extension relationships for a Swift/ObjC file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `string` | Absolute path to the file to analyze |

**Returns:**
- `upstream`: Classes/protocols this file inherits from or conforms to
- `downstream`: Classes that inherit from or depend on symbols in this file
- `extensions`: Files that extend types defined in this file

---

### `read_symbol`
Read the **live source code** for a specific symbol using its byte range.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `string` | Absolute path to the source file |
| `start_byte` | `int` | Start byte offset |
| `end_byte` | `int` | End byte offset |

**USE THIS INSTEAD OF:** `cat`, `head`, `tail`, or reading entire files.

---

### `find_bridging_header_usage`
Find all Swift classes that inherit from Objective-C classes.

**Returns:** Count and details of Swift/ObjC bridging relationships.

---

## Setup & Configuration

See the full setup guide: **[engine/CONNECTION_GUIDE.md](engine/CONNECTION_GUIDE.md)**

### Quick Start

1.  **Install dependencies:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cd ~/tools/iOS-GraphRAG
    uv init && uv venv --python 3.11
    source .venv/bin/activate
    uv pip install torch torchvision
    uv pip install "mcp[cli]" tree-sitter tree-sitter-swift tree-sitter-objc networkx numpy sentence-transformers tqdm
    ```

2.  **Build the knowledge graph:**
    ```bash
    uv run ios-graphrag-index --repo /path/to/your/ios-project
    ```

3.  **Register with your AI client** (Claude Desktop example):
    ```json
    {
      "mcpServers": {
        "iOS-GraphRAG": {
          "command": "/Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/ios-graphrag-server",
          "env": {
            "GRAPH_DB_PATH": "/Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite"
          }
        }
      }
    }
    ```

---

## Example Usage

### 1. Semantic Code Discovery

> **Prompt:** "Find code related to user authentication"

The AI will call:
```
semantic_search(query="user authentication login session management")
```

**Result:** Top 10 files/symbols semantically related to authentication, ranked by similarity score.

---

### 2. Understanding Dependencies

> **Prompt:** "What depends on UserManager.swift?"

The AI will call:
```
trace_dependencies(file_path="/path/to/UserManager.swift")
```

**Result:**
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

---

### 3. Reading Specific Code

> **Prompt:** "Show me the implementation of that function"

After `semantic_search` returns byte ranges, the AI calls:
```
read_symbol(
  file_path="/path/to/AuthService.swift",
  start_byte=1024,
  end_byte=2048
)
```

**Result:** The exact current source code from disk — never stale.

---

### 4. Swift/ObjC Interoperability Audit

> **Prompt:** "Find all Swift classes that inherit from Objective-C"

The AI will call:
```
find_bridging_header_usage()
```

**Result:**
```json
{
  "count": 42,
  "bridging_classes": [
    {
      "swift_class": "ModernViewController",
      "swift_file": "/path/to/ModernViewController.swift",
      "objc_parent": "LegacyBaseController",
      "objc_file": "/path/to/LegacyBaseController.h"
    }
  ]
}
```

---

## Performance Targets

| Operation | Target |
|-----------|--------|
| Full index (1M+ lines) | < 30 minutes |
| Incremental re-index (10 files) | < 10 seconds |
| `trace_dependencies()` | < 100ms |
| `semantic_search()` | < 500ms |

---

## Hardware Optimization

Optimized for **Apple Silicon (M-series)**:

- **Parsing:** Parallel via `ProcessPoolExecutor` (saturates all cores)
- **Embeddings:** `nomic-embed-text-v1.5` on MPS (Metal Performance Shaders)
- **Graph Traversal:** NetworkX in-memory for O(1) lookups
- **Storage:** SQLite with targeted indexes

---

## License

MIT License — see [LICENSE](LICENSE)
