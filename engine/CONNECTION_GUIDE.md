# Setup Guide: The "Last Mile" Connection

The SQLite database is inert on its own. The MCP Server (your Python script) is the bridge. You must "register" this script in your AI client's configuration file.

> [!IMPORTANT]
> **This is a `stdio`-based MCP server.** You do NOT need to run a background process. The AI client (Copilot CLI, Claude Desktop, VS Code) spawns the Python process automatically when it starts and manages its lifecycle. Your job is just to point the config at the right Python binary and database file.

---

## Step 0: Install uv (If needed)

We use `uv` for high-performance Python package management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Step 1: Environment Setup

1. **Create a tools directory:**
   ```bash
   mkdir -p ~/tools/iOS-GraphRAG
   cd ~/tools/iOS-GraphRAG
   ```

2. **Initialize Environment:**
   ```bash
   uv init
   uv venv --python 3.11
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   > [!IMPORTANT]
   > For M3 Max/Apple Silicon, we install a specific PyTorch build first to ensure clean MPS (Metal Performance Shaders) support.

   ```bash
   # Install PyTorch (Standard PyPI wheels include MPS support for Mac)
   uv pip install torch torchvision

   # Install MCP and Tooling
   uv pip install "mcp[cli]" tree-sitter tree-sitter-swift tree-sitter-objc networkx numpy sentence-transformers tqdm
   ```

---

## Step 2: Build the Graph

Run the indexer pointing to your iOS repository:

```bash
GRAPH_DB_PATH=/absolute/path/to/output/knowledge-graph.sqlite \
  uv run indexer_prod.py --repo /path/to/your/ios-project
```

**Result:** A `knowledge-graph.sqlite` file will be created. Note its **absolute path** — you'll need it for Step 3.

---

## Step 3: Register the Server

The config always needs two absolute paths:
- **Python binary:** the `.venv/bin/python` inside your tools directory
- **SQLite database:** the `knowledge-graph.sqlite` file built in Step 2

### Option A: GitHub Copilot CLI

Add to `~/.config/gh-copilot/config.yml` (or wherever `gh copilot` reads its MCP config):

```yaml
# ~/.config/gh-copilot/config.yml
mcp_servers:
  iOS-GraphRAG:
    command: /Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/python
    args:
      - /Users/YOUR_USER/tools/iOS-GraphRAG/engine/core/server_prod.py
    env:
      GRAPH_DB_PATH: /Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite
```

> [!NOTE]
> GitHub Copilot CLI uses **stdio transport** — the Python process is spawned and killed automatically by `gh copilot`. You do not run a background server.

### Option B: GitHub Copilot in VS Code

Add to your VS Code `settings.json` (open via ⌘+Shift+P → "Open User Settings JSON"):

```json
"github.copilot.chat.mcp.servers": {
    "iOS-GraphRAG": {
        "command": "/Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/python",
        "args": ["/Users/YOUR_USER/tools/iOS-GraphRAG/engine/core/server_prod.py"],
        "env": {
            "GRAPH_DB_PATH": "/Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite"
        }
    }
}
```

### Option C: Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "iOS-GraphRAG": {
      "command": "/Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/python",
      "args": [
        "/Users/YOUR_USER/tools/iOS-GraphRAG/engine/core/server_prod.py"
      ],
      "env": {
        "GRAPH_DB_PATH": "/Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite"
      }
    }
  }
}
```

---

## Step 4: Verification

Before using it in chat, verify the toolchain with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector \
  /Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/python \
  /Users/YOUR_USER/tools/iOS-GraphRAG/engine/core/server_prod.py
```

1. Select the `trace_dependencies` tool.
2. Enter a known file path.
3. If you see valid JSON architecture data, you're ready.

---

## Troubleshooting

### "Cannot send a request, as the client has been closed"

This is the most common error when setting up on a new machine. It means the **server Python process crashed during startup** before it could complete the MCP stdio handshake. The Copilot CLI sees a dead process and reports "client closed."

**How to diagnose:** After the first failed attempt, check the log file the server writes alongside the database:

```bash
cat /path/to/your/knowledge-graph-directory/server_prod.log
```

Common causes and fixes:

| Symptom in log | Fix |
|---|---|
| `FATAL: knowledge-graph.sqlite not found` | The `GRAPH_DB_PATH` env var is missing or wrong in your config. Use the **absolute path** to the `.sqlite` file. |
| `ModuleNotFoundError: No module named 'mcp'` | The `command` in your config points to the wrong Python. It must point to `.venv/bin/python`, not the system Python. |
| `SQLite error: no such table: nodes` | The DB path points to a different/corrupt file. Re-run `indexer_prod.py`. |
| SSL errors / `certificate verify failed` | See "Enterprise SSL" section below. |

### Enterprise SSL (Work PC / Corporate Proxy)

On a work machine with a corporate SSL proxy, `sentence_transformers` may fail downloading model weights. The fix is to **pre-download the model at home** and point the server at a local cache:

```bash
# Run once (on a machine without proxy):
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
print('Model cached.')
"
```

Then copy the model cache to your work machine and set the environment variable:

```json
"env": {
    "GRAPH_DB_PATH": "/absolute/path/to/knowledge-graph.sqlite",
    "SENTENCE_TRANSFORMERS_HOME": "/absolute/path/to/model/cache"
}
```

Alternatively, `server_prod.py` already calls `ensure_model()` lazily (only on first `semantic_search` call), so if you avoid that tool on first run, the server will start and the other tools (`trace_dependencies`, `read_symbol`) will work without any model.

### Other Issues

- **Stale Data:** If you do a massive refactor, re-run `indexer_prod.py` to refresh the graph.
- **MPS Not Detected:** Ensure you followed the specific PyTorch install step in Step 1.
- **Wrong Python version:** The server requires Python 3.11+. Verify with `/path/to/.venv/bin/python --version`.
