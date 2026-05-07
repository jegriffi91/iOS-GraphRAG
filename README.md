# iOS-GraphRAG

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml) <!-- TODO: replace OWNER/REPO with the actual GitHub path -->

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

## Symbol Types

The `nodes.symbol_type` column tags every indexed symbol. The current set
(constrained by a CHECK in migration 004):

- `class` — Swift `class` declarations
- `struct` — Swift `struct` declarations
- `protocol` — Swift `protocol` declarations
- `enum` — Swift `enum` declarations
- `function` — Swift `func` declarations (free functions and methods)
- `extension` — Swift `extension` blocks
- `property` — stored properties (`var`/`let`), including `lazy` and `static`
- `computed_property` — properties with a getter/setter block
- `initializer` — `init` declarations (plain, failable, convenience, required)
- `deinitializer` — `deinit` blocks

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

**Returns** (orientation matches LSP's `findReferences` / `prepareCallHierarchy`):
- `upstream`: Symbols this file depends on — classes/protocols it inherits from or conforms to, and functions it calls
- `downstream`: Symbols that depend on this file — subclasses, protocol conformers, and callers of its symbols
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

> **Prompt:** "What does Calculator.swift inherit from, and what depends on it?"

The AI will call:
```
trace_dependencies(file_path="/path/to/Calculator.swift")
```

**Result** (Calculator inherits from BaseCalculator; ScientificCalculator and BasicCalculator inherit from Calculator):
```json
{
  "target": {
    "path": "/path/to/Calculator.swift",
    "symbols": ["Calculator"]
  },
  "upstream": [
    {"path": "/path/to/BaseCalculator.swift", "symbol": "BaseCalculator", "edge_type": "INHERITS"}
  ],
  "downstream": [
    {"path": "/path/to/ScientificCalculator.swift", "symbol": "ScientificCalculator", "edge_type": "INHERITS"},
    {"path": "/path/to/BasicCalculator.swift", "symbol": "BasicCalculator", "edge_type": "INHERITS"}
  ],
  "extensions": [
    "/path/to/Calculator+Memory.swift"
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

## CI

On every PR, CI runs ruff lint + format check, pytest on macOS Python 3.11/3.12, and `ios-graphrag-preflight --smoke` (offline checks only — the full preflight pulls a ~500MB embedding model and is meant for a fresh local machine, not CI).

A nightly workflow runs the benchmark harness against the bundled `test_fixtures/CalculatorApp` fixture and uploads the JSON as a build artifact (90-day retention).

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and [`.github/workflows/nightly-benchmark.yml`](.github/workflows/nightly-benchmark.yml) for the full workflow definitions.

---

## Operations

- Release notes: [`CHANGELOG.md`](CHANGELOG.md)
- On-call runbook: [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- Bug-report flow: [`docs/REPORTING.md`](docs/REPORTING.md)

---

## License

MIT License — see [LICENSE](LICENSE)
