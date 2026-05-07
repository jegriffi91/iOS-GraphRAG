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

For automatic incremental reindex on file changes, see [`dist/launchd/README.md`](../dist/launchd/README.md) for the watcher daemon.

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

`sentence_transformers` needs to download an embedding model from Hugging Face. If your corporate proxy intercepts TLS with a self-signed certificate, the download fails with `certificate verify failed`.

#### Recommended: point at your corporate CA bundle

Ask your security team for the path to the corporate CA bundle (a PEM file). Both the indexer and the server accept it directly:

```bash
# CLI flag — preferred for one-off runs
ios-graphrag-index --repo /path/to/repo --cert-bundle /path/to/corp-bundle.pem

# Same flag on the server (typically set in your MCP client config)
ios-graphrag-server --cert-bundle /path/to/corp-bundle.pem
```

Or set it once for the shell:

```bash
export REQUESTS_CA_BUNDLE=/path/to/corp-bundle.pem
export SSL_CERT_FILE=/path/to/corp-bundle.pem
```

In your client config (Copilot CLI / VS Code / Claude Desktop), pass the flag through `args` and/or set the env vars. Example for Claude Desktop:

```json
{
  "mcpServers": {
    "iOS-GraphRAG": {
      "command": "/Users/YOUR_USER/tools/iOS-GraphRAG/.venv/bin/ios-graphrag-server",
      "args": ["--cert-bundle", "/Users/YOUR_USER/security/corp-bundle.pem"],
      "env": {
        "GRAPH_DB_PATH": "/Users/YOUR_USER/tools/iOS-GraphRAG/knowledge-graph.sqlite"
      }
    }
  }
}
```

TLS verification stays **on** — the cert is verified against your corporate CA. This is what passes infosec review.

#### Last resort: disable TLS verification entirely

If your environment has no available corporate CA bundle and the security team explicitly approves an interim bypass, set the opt-in env var:

```bash
export GRAPHRAG_INSECURE_TLS=1
```

> [!CAUTION]
> **Get security team sign-off before using this.** It bypasses TLS verification across all outbound HTTPS calls (including Hugging Face model downloads), which means a man-in-the-middle attacker on your network could swap in malicious model weights. Prefer the cert-bundle approach above.

When this env var is set, the indexer/server log a `WARNING: TLS verification disabled by GRAPHRAG_INSECURE_TLS=1; ...` line on startup so the bypass is visible in audit logs.

The change to make this behavior opt-in landed in Phase 2 of the production hardening roadmap (see `docs/PRODUCTION_ROADMAP.md` §3 → Phase 2). Pre-Phase-2 builds bypassed TLS unconditionally at module import; if you're upgrading from one of those, your existing flows still work but you may now hit `certificate verify failed` until you set `--cert-bundle` or `GRAPHRAG_INSECURE_TLS=1`.

#### Pre-cache the model offline

Independent of TLS, you can avoid the runtime download altogether by caching the model on a machine with internet access and copying the cache:

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

### Offline / Pre-staged Model

Some corporate networks block Hugging Face entirely, even with the corp CA bundle in place. In that case, stage the embedding model on a machine that has internet access, distribute the resulting directory through your usual internal channels, and point the indexer at the staged copy. The runtime never reaches out to `huggingface.co`.

#### Stage the model

On a machine with internet access:

```bash
uv run pip install huggingface_hub
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 \
    --local-dir ~/.cache/ios-graphrag/models/nomic-embed-text-v1.5
```

The resulting directory contains `config.json`, `model.safetensors`, the tokenizer files, and `1_Pooling/`. Roughly 500 MB on disk.

#### Distribute

Ship the `nomic-embed-text-v1.5/` directory to teammates' `~/.cache/ios-graphrag/models/` via whatever your team already uses:

- `rsync` / `scp` for one-off transfers between dev machines.
- An internal artifact server (Artifactory, S3, internal package registry) for self-serve pull. <!-- TODO: org-specific Artifactory/S3 URL -->
- A shared network drive that's already trusted by IT.

#### Override the cache location

If the cache directory is non-default — for example a shared `/opt/models/` mount, or the model lives next to your project under `./vendor/models/` — point the indexer at it explicitly:

```bash
export IOS_GRAPHRAG_MODEL_DIR=/opt/models/nomic-embed-text-v1.5
```

Resolution order at indexer startup:

1. `$IOS_GRAPHRAG_MODEL_DIR` if set and populated with weight files.
2. `~/.cache/ios-graphrag/models/nomic-embed-text-v1.5/` if populated.
3. Hugging Face cache / network download (existing behavior).

The indexer logs which path it loaded the model from (`model loaded from ...`) so you can confirm the override took effect.

#### Verify

```bash
uv run ios-graphrag-preflight
```

Should report the model loaded successfully without any network access. If preflight reports the model as missing, double-check the directory contains both `config.json` and `model.safetensors` (or `pytorch_model.bin`).

### Quick Fixes

| Problem | Fix |
|---|---|
| Stale data after a big refactor | Re-run `ios-graphrag-index` |
| MPS (Metal) not detected | Reinstall PyTorch per Step 1 |
| Wrong Python version | Server needs 3.11+. Check: `.venv/bin/python --version` |

For filing bug reports (crashes, recoverable tool errors, general health
checks), see `docs/REPORTING.md` for the diagnostic-paste workflow.
