# iOS-GraphRAG: Setup Guide

> [!IMPORTANT]
> **This is a `stdio` MCP server.** You do NOT run it yourself. Your AI client spawns and kills the process automatically. You just point a config file at two paths: (1) the Python binary, (2) the database file.

---

## Step 0 — Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Step 1 — Create the Python Environment

```bash
mkdir -p ~/tools/iOS-GraphRAG && cd ~/tools/iOS-GraphRAG

uv init
uv venv --python 3.11
source .venv/bin/activate

# PyTorch first (includes MPS/Metal support on Apple Silicon)
uv pip install torch torchvision

# Everything else
uv pip install "mcp[cli]" tree-sitter tree-sitter-swift tree-sitter-objc \
  networkx numpy sentence-transformers tqdm
```

---

## Step 2 — Build the Graph

```bash
GRAPH_DB_PATH=/absolute/path/to/knowledge-graph.sqlite \
  uv run ios-graphrag-index --repo /path/to/your/ios-project
```

This creates `knowledge-graph.sqlite`. **Write down its absolute path** — you need it next.

---

## Step 3 — Register the Server

Every config needs exactly **two absolute paths**. Replace `YOUR_USER` with your macOS username.

| Path | Value |
|---|---|
| **Python binary** | `/Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/python` |
| **Database file** | `/Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite` |

> [!CAUTION]
> **All paths MUST be absolute.** No `~`, no `$HOME`, no relative paths. Every client silently fails on anything but a full `/Users/…` path.

Pick your client below and paste the config:

### GitHub Copilot CLI

File: `~/.config/gh-copilot/config.yml`

```yaml
mcp_servers:
  iOS-GraphRAG:
    command: /Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/ios-graphrag-server
    env:
      GRAPH_DB_PATH: /Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite
```

### GitHub Copilot in VS Code

Open via **⌘+Shift+P → "Open User Settings JSON"** and add:

```json
"github.copilot.chat.mcp.servers": {
    "iOS-GraphRAG": {
        "command": "/Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/ios-graphrag-server",
        "env": {
            "GRAPH_DB_PATH": "/Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite"
        }
    }
}
```

### Claude Desktop

File: `~/Library/Application Support/Claude/claude_desktop_config.json`

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

## Step 4 — Verify It Works

```bash
npx @modelcontextprotocol/inspector \
  /Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/ios-graphrag-server
```

1. Click the `trace_dependencies` tool.
2. Enter any file path from your iOS project.
3. **See valid JSON?** You're done. 🎉

---

## Troubleshooting

### "Cannot send a request, as the client has been closed"

**Translation:** the Python process crashed before it could say hello. Check the log:

```bash
cat /path/to/your/knowledge-graph-directory/server.log
```

| What the log says | What you did wrong | Fix |
|---|---|---|
| `FATAL: knowledge-graph.sqlite not found` | `GRAPH_DB_PATH` is wrong or missing | Use the **absolute** path to the `.sqlite` file |
| `ModuleNotFoundError: No module named 'mcp'` | `command` points to system Python | Change it to `.venv/bin/python` |
| `no such table: nodes` | DB is corrupt or wrong file | Re-run `ios-graphrag-index` |
| SSL / `certificate verify failed` | Corporate proxy blocking downloads | See **Enterprise SSL** below |

### Enterprise SSL (Corporate Proxy)

`sentence_transformers` needs to download a model. If your corporate proxy blocks it:

1. **On a home machine** (no proxy), cache the model:
   ```bash
   python -c "
   from sentence_transformers import SentenceTransformer
   SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
   print('Cached.')
   "
   ```
2. **Copy** the cache folder to your work machine.
3. **Add** `SENTENCE_TRANSFORMERS_HOME` to your config's `env`:
   ```json
   "env": {
       "GRAPH_DB_PATH": "/absolute/path/to/knowledge-graph.sqlite",
       "SENTENCE_TRANSFORMERS_HOME": "/absolute/path/to/model/cache"
   }
   ```

> [!TIP]
> No proxy workaround needed for `trace_dependencies` or `read_symbol` — only `semantic_search` uses the model. Skip the model download entirely if you don't need semantic search.

### Quick Fixes

| Problem | Fix |
|---|---|
| Stale data after a big refactor | Re-run `ios-graphrag-index` |
| MPS (Metal) not detected | Reinstall PyTorch per Step 1 |
| Wrong Python version | Server needs 3.11+. Check: `.venv/bin/python --version` |
