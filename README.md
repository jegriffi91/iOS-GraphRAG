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
(constrained by a CHECK in migration 006):

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
- `enum_case` — Swift `case` declarations inside `enum` bodies (bare,
  raw-valued, and associated-value forms; multi-case-per-line `case a, b, c`
  produces one row per case). Cases with associated values get a call-site
  selector like `custom(name:fn:)`; bare and raw-valued cases have NULL
  selector.
- `subscript` — Swift `subscript(...)` declarations. `symbol_name` is the
  literal `subscript`; overloads are disambiguated via the selector column
  (e.g. `subscript(key:)`, `subscript(index:withDefault:)`).
- `typealias` — Swift `typealias` declarations (file scope or nested).
  `selector` is NULL because typealiases are not invocable.
- `objc_class` — Objective-C `@interface`/`@implementation` class
  declarations. The header and implementation files each contribute one row.
- `objc_method` — Objective-C method declarations and definitions
  (both `-` instance and `+` class methods). `symbol_name` and `selector`
  carry the canonical selector form (e.g. `addValue:toValue:`,
  `sharedInstance`).
- `objc_property` — Objective-C `@property` declarations.
- `objc_protocol` — Objective-C `@protocol` declarations.
- `category` — Objective-C category declarations (`@interface
  CalculatorMath (Trig)`). `symbol_name` is the parenthesized category
  name; the host class is captured separately as `objc_class`.

### SwiftUI enrichment columns

Migration 002 adds four nullable columns (populated for Swift symbols by
the Phase 4.5d extractor) for SwiftUI- and Combine-aware querying:

- `is_swiftui_view` (`BOOLEAN`) — set on struct/class declarations whose
  inheritance list includes `View`, `ViewModifier`, or any name ending
  in `View`.
- `is_observable` (`BOOLEAN`) — set on class declarations annotated with
  `@Observable` (Swift Observation) or `@Model` (SwiftData). Note:
  `@Published` is property-level and does NOT flip this column;
  `@Published` is captured per-property via `state_kind` instead.
- `state_kind` (`TEXT`) — set on properties decorated with a known
  reactivity wrapper. Values: `state`, `binding`, `stateobject`,
  `observedobject`, `environmentobject`, `environment`, `appstorage`,
  `scenestorage`, `fetchrequest`, `query`, `published`, `focusstate`,
  `bindable`, `gesturestate`, `scaledmetric`, `namespace`,
  `accessibilityfocusstate`. When multiple wrappers are stacked, the
  FIRST matching wrapper in source order wins.
- `body_kind` (`TEXT`) — set to `viewbody` on the SwiftUI `body: some View`
  computed property and to `resultbuilder` on `@ViewBuilder` functions
  / computed properties. NULL on regular functions and stored
  properties.

The `find_swiftui_views` MCP tool surfaces these columns -- see below.

---

## MCP Tools

The server exposes five tools to AI clients via the Model Context Protocol:

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

### `find_swiftui_views`
List every SwiftUI `View` (`is_swiftui_view=1`) along with the
state-binding kinds it declares (`@State`, `@StateObject`, `@Binding`, etc.).
Backed by the migration-002 SwiftUI enrichment columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_kind` | `string?` | Optional. Restrict to Views with at least one property whose `state_kind` matches (e.g. `'stateobject'`). |
| `observable_only` | `bool` | Optional, default `False`. Restrict to Views in files that declare at least one `@Observable` / `@Model` class (file-locality approximation). |

**Returns:** Count and per-View list of `{swift_class, file_path, states}`.

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

A nightly workflow runs the benchmark harness against the bundled `test_fixtures/CalculatorApp` fixture and uploads the JSON as a build artifact (90-day retention). Nightly CI compares against `benchmarks/baseline/` and fails the run on a wall-time, semantic-search-p99, or parse-error regression past the tolerances in `benchmarks/check_regression.py`.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and [`.github/workflows/nightly-benchmark.yml`](.github/workflows/nightly-benchmark.yml) for the full workflow definitions.

---

## Operations

- Release notes: [`CHANGELOG.md`](CHANGELOG.md)
- On-call runbook: [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- Bug-report flow: [`docs/REPORTING.md`](docs/REPORTING.md)

---

## License

MIT License — see [LICENSE](LICENSE)
