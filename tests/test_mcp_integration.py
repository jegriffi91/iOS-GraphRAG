"""End-to-end MCP integration tests over stdio JSON-RPC.

The other 161 tests in this suite exercise tool *handlers* directly
(``server._semantic_search(...)``, etc.) but never go through the
MCP framing layer. A regression in ``tools/list`` shape, ``tools/call``
argument plumbing, or response serialization would slip past every one
of those tests.

This file spawns ``ios-graphrag-server`` as a subprocess against the
indexed test fixture and speaks MCP JSON-RPC over its stdin/stdout. The
tests cover the five registered tools' wire contract: initialize
handshake, tools listing, success and error envelopes (incl. that no
Python traceback leaks into the response), trace_id propagation, and
clean shutdown on stdin close.

Wire format note: despite the spec snippet in the original task brief
referencing Content-Length framing, the MCP stdio transport (both
client and server sides in this version of the ``mcp`` package) uses
*newline-delimited JSON* — see ``mcp/server/stdio.py`` and
``mcp/client/stdio/__init__.py``. We frame the same way here.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Generator, Tuple

import pytest

# ---------------------------------------------------------------------------
# Helpers: minimal newline-delimited JSON-RPC client over a Popen subprocess.
# ---------------------------------------------------------------------------


def _send(proc: subprocess.Popen, message: dict) -> None:
    """Write a single JSON-RPC frame to the server's stdin.

    MCP stdio is one JSON object per line, terminated by ``\\n`` (no
    Content-Length headers).
    """
    line = (json.dumps(message) + "\n").encode("utf-8")
    assert proc.stdin is not None
    proc.stdin.write(line)
    proc.stdin.flush()


def _recv_line(proc: subprocess.Popen, timeout: float = 30.0) -> str:
    """Read one ``\\n``-terminated line from the server's stdout.

    Uses ``select`` against ``proc.stdout.fileno()`` so we never block
    forever on a hung server. Returns the decoded line WITHOUT the
    trailing newline. Raises ``TimeoutError`` or ``EOFError`` rather than
    hanging the test session.
    """
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout
    buffer = b""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"no MCP response within {timeout}s; partial buffer: {buffer!r}"
            )
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            raise EOFError(
                f"server closed stdout before sending newline; partial: {buffer!r}"
            )
        buffer += chunk
        if b"\n" in buffer:
            line, _, rest = buffer.partition(b"\n")
            if rest:
                # MCP stdio: server SHOULD send one frame at a time, but
                # if multiple frames arrived in one read we'd lose them.
                # In practice this never happens for our request/response
                # pattern; surface as a failure rather than silently drop.
                raise RuntimeError(
                    f"unexpected trailing bytes after newline: {rest!r}"
                )
            return line.decode("utf-8")


def _recv_response(
    proc: subprocess.Popen, expected_id: int, timeout: float = 30.0
) -> dict:
    """Read frames until we see one matching ``expected_id``.

    The server may interleave notifications (no ``id``) — we skip those
    and return only the matching response.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining == 0.0:
            raise TimeoutError(
                f"no response for id={expected_id} within {timeout}s"
            )
        line = _recv_line(proc, timeout=remaining)
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"server emitted malformed JSON-RPC frame: {line!r} ({exc})"
            ) from exc
        if msg.get("id") == expected_id:
            return msg
        # Notifications (no id) and responses for other ids are ignored.


def _wait_for_ready(proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Wait until the server logs the 'READY' marker on stderr.

    server.py emits ``log.info("ios-graphrag-server READY")`` immediately
    before ``mcp.run()`` blocks on stdio. Waiting for that line is more
    reliable than a fixed sleep — the model load and graph load can take
    multiple seconds the first time the test session runs.
    """
    assert proc.stderr is not None
    fd = proc.stderr.fileno()
    deadline = time.monotonic() + timeout
    buffer = b""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited (rc={proc.returncode}) before READY: "
                f"{buffer.decode('utf-8', errors='replace')}"
            )
        ready, _, _ = select.select([fd], [], [], 0.5)
        if not ready:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            raise EOFError(
                "server closed stderr before READY: "
                + buffer.decode("utf-8", errors="replace")
            )
        buffer += chunk
        if b"ios-graphrag-server READY" in buffer:
            return
    raise TimeoutError(
        f"server did not emit READY within {timeout}s; "
        f"stderr so far: {buffer.decode('utf-8', errors='replace')}"
    )


# ---------------------------------------------------------------------------
# Session-scoped fixture: spawn the server once, reuse across all 7 tests.
# Per-test spawn would re-pay model load + graph load every time (~5s+).
# ---------------------------------------------------------------------------


# Shape: (proc, send_fn, recv_fn, request_id_counter)
_McpClient = Tuple[
    subprocess.Popen,
    Callable[[str, dict | None], dict],
    Callable[[], int],
]


@pytest.fixture(scope="session")
def mcp_server(
    indexed_db_path: Path, tmp_path_factory
) -> Generator[_McpClient, None, None]:
    """Spawn ios-graphrag-server, complete the initialize handshake, yield a client.

    The fixture:
      * Pins ``GRAPH_DB_PATH`` at the indexed test DB.
      * Pins ``GRAPHRAG_LOG_DIR`` at a tmpdir so logs don't leak into
        ``~/Library/Logs/ios-graphrag/``.
      * Waits for the ``READY`` marker on stderr before the first send.
      * Performs the ``initialize`` handshake + ``initialized``
        notification so subsequent ``tools/list`` / ``tools/call``
        requests aren't blocked by the session state machine.
      * On teardown: closes stdin, waits up to 10s for clean exit, then
        kills the process if it is still alive.
    """
    log_dir = tmp_path_factory.mktemp("ios-graphrag-mcp-integration-logs")
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(indexed_db_path)
    env["GRAPHRAG_LOG_DIR"] = str(log_dir)

    proc = subprocess.Popen(
        [sys.executable, "-m", "ios_graphrag.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )

    next_id = 0

    def _next_id() -> int:
        nonlocal next_id
        next_id += 1
        return next_id

    def _request(method: str, params: dict | None = None) -> dict:
        rid = _next_id()
        frame: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            frame["params"] = params
        _send(proc, frame)
        return _recv_response(proc, expected_id=rid)

    try:
        _wait_for_ready(proc)

        # MCP initialize handshake. Use the latest version this client
        # build of mcp ships; the server will negotiate down if needed.
        init_resp = _request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.0.0"},
            },
        )
        if "error" in init_resp:
            pytest.fail(f"initialize failed: {init_resp['error']}")

        # Per-spec: client must send initialized notification before
        # making non-initialize requests.
        _send(
            proc,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

        yield proc, _request, _next_id
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        # Drain remaining stdout/stderr so the OS doesn't hold pipe buffers.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initialize_returns_server_info(indexed_db_path: Path, tmp_path_factory):
    """The MCP initialize response carries protocolVersion, serverInfo,
    and capabilities.tools.

    This test deliberately does NOT use the session fixture: that fixture
    has already consumed the initialize handshake. Spawning a separate
    short-lived server here is the only way to inspect the raw
    initialize response. We use a lightweight in-place spawn so the cost
    is paid exactly once.
    """
    log_dir = tmp_path_factory.mktemp("ios-graphrag-mcp-integration-init-logs")
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(indexed_db_path)
    env["GRAPHRAG_LOG_DIR"] = str(log_dir)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ios_graphrag.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    try:
        _wait_for_ready(proc)
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "0.0.0"},
                },
            },
        )
        resp = _recv_response(proc, expected_id=1)

        assert "error" not in resp, f"initialize returned error: {resp}"
        assert resp.get("jsonrpc") == "2.0"
        result = resp["result"]
        assert "protocolVersion" in result
        assert isinstance(result["protocolVersion"], (str, int))

        server_info = result.get("serverInfo")
        assert server_info is not None, f"missing serverInfo: {result}"
        assert server_info.get("name") == "iOS-GraphRAG"

        capabilities = result.get("capabilities")
        assert capabilities is not None
        assert "tools" in capabilities, (
            f"server did not advertise tools capability: {capabilities}"
        )
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


def test_tools_list_returns_five_tools(mcp_server: _McpClient):
    """tools/list must return all 5 registered tools, each with an inputSchema."""
    _proc, request, _ = mcp_server
    resp = request("tools/list", {})
    assert "error" not in resp, f"tools/list returned error: {resp}"
    tools = resp["result"]["tools"]

    names = {tool["name"] for tool in tools}
    expected = {
        "global_codebase_search",
        "swift_dependency_tracer",
        "read_symbol_source",
        "objc_swift_bridge_finder",
        "find_swiftui_views",
    }
    assert expected.issubset(names), (
        f"missing expected tools: {expected - names}; got: {names}"
    )

    for tool in tools:
        assert "inputSchema" in tool, f"tool {tool['name']!r} missing inputSchema"
        schema = tool["inputSchema"]
        assert isinstance(schema, dict), f"inputSchema not a dict: {schema!r}"
        # JSON Schema convention: top-level type is 'object' for tool args.
        assert schema.get("type") == "object", (
            f"tool {tool['name']!r} inputSchema.type != 'object': {schema!r}"
        )


def test_tools_call_global_codebase_search_returns_results(mcp_server: _McpClient):
    """tools/call for global_codebase_search returns a CallToolResult with
    structured content (no isError, no traceback)."""
    _proc, request, _ = mcp_server
    resp = request(
        "tools/call",
        {
            "name": "global_codebase_search",
            "arguments": {"query": "Calculator", "top_k": 3},
        },
    )
    assert "error" not in resp, f"tools/call returned protocol error: {resp}"

    result = resp["result"]
    # Successful call: isError is False or absent.
    assert not result.get("isError", False), (
        f"global_codebase_search returned isError=True: {result}"
    )

    # FastMCP serializes dict returns into both ``content`` (text JSON)
    # and ``structuredContent`` (the dict itself). Either path must
    # contain non-empty results, since the indexed fixture has
    # Calculator-related symbols.
    structured = result.get("structuredContent")
    if structured is not None:
        assert "results" in structured, f"structuredContent missing results: {structured}"
        assert isinstance(structured["results"], list)
        assert len(structured["results"]) > 0, (
            "no Calculator-related symbols in structured response — "
            "either embeddings broke or the fixture changed"
        )
    else:
        # Fallback: parse the text content payload.
        content = result.get("content", [])
        assert content, "no content in successful call result"
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert text_blocks, f"no text content blocks: {content}"
        parsed = json.loads(text_blocks[0]["text"])
        assert "results" in parsed
        assert len(parsed["results"]) > 0


def test_tools_call_unknown_tool_returns_mcp_error(mcp_server: _McpClient):
    """tools/call for a tool that doesn't exist must return an MCP-shaped
    error (CallToolResult with isError=True), NOT a JSON-RPC protocol
    error and NOT a Python traceback leaking through.

    Traceback leakage is a security concern: it can expose internal paths
    and code structure to a misbehaving client.
    """
    _proc, request, _ = mcp_server
    resp = request(
        "tools/call",
        {"name": "does_not_exist", "arguments": {}},
    )
    # The mcp lowlevel server catches ToolError("Unknown tool: ...")
    # and converts it into an isError=True CallToolResult.
    assert "result" in resp, f"unknown tool produced no result: {resp}"
    result = resp["result"]
    assert result.get("isError") is True, (
        f"unknown tool did not return isError=True: {result}"
    )

    # Stitch together all text content for a single search target.
    content = result.get("content", [])
    text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
    assert text, f"isError result has no text content: {result}"

    # Crucially, no Python traceback leak:
    assert "Traceback (most recent call last)" not in text, (
        f"Python traceback leaked into MCP response: {text!r}"
    )
    assert 'File "' not in text, (
        f"file path / source line leaked into MCP response: {text!r}"
    )


def test_tools_call_invalid_args_returns_mcp_error(mcp_server: _McpClient):
    """tools/call for global_codebase_search with a non-int top_k must
    return an MCP error envelope (isError=True), not crash the server.

    The lowlevel server validates ``arguments`` against ``inputSchema``
    via jsonschema before dispatching. ``top_k`` is constrained
    ``{type: integer, ge: 1, le: 50}`` by the Pydantic Field annotation,
    so a string fails validation cleanly.
    """
    _proc, request, _ = mcp_server
    resp = request(
        "tools/call",
        {
            "name": "global_codebase_search",
            "arguments": {"query": "Calculator", "top_k": "not_a_number"},
        },
    )
    assert "result" in resp, f"invalid args produced no result: {resp}"
    result = resp["result"]
    assert result.get("isError") is True, (
        f"invalid args did not return isError=True: {result}"
    )

    content = result.get("content", [])
    text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
    assert text, f"isError result has no text content: {result}"

    # No traceback leakage on the validation path either.
    assert "Traceback (most recent call last)" not in text, (
        f"validation error leaked Python traceback: {text!r}"
    )


def test_trace_id_present_in_error_response(mcp_server: _McpClient):
    """Phase 6b contract: when a tool handler returns a structured
    error payload, the @_traced_handler decorator injects a trace_id
    so the client can grep server.log for the same id.

    We exercise read_symbol_source with a path that doesn't exist on
    disk. The handler returns ``{"error": "FILE_DELETED", ...}`` (does
    NOT raise). The decorator surfaces ``trace_id`` into the dict
    response, which FastMCP serializes as structuredContent.
    """
    _proc, request, _ = mcp_server
    bogus_path = "/nonexistent/path/to/some/file.swift"
    resp = request(
        "tools/call",
        {
            "name": "read_symbol_source",
            "arguments": {
                "file_path": bogus_path,
                "start_byte": 0,
                "end_byte": 10,
            },
        },
    )
    assert "result" in resp, f"read_symbol_source produced no result: {resp}"
    result = resp["result"]
    # The handler returns a normal dict (not an exception), so isError
    # is False — the error semantics live INSIDE the dict's ``error``
    # field. trace_id is injected as a sibling key.
    assert not result.get("isError", False), (
        f"read_symbol_source unexpectedly raised: {result}"
    )

    # trace_id should be on the structuredContent payload (the dict the
    # handler returned, with the decorator's setdefault applied).
    structured = result.get("structuredContent")
    text_blocks = result.get("content") or []
    text_payload = None
    if structured is not None:
        payload = structured
    else:
        text_payload = next(
            (b.get("text") for b in text_blocks if b.get("type") == "text"),
            None,
        )
        assert text_payload, f"no payload to inspect: {result}"
        payload = json.loads(text_payload)

    assert "trace_id" in payload, (
        f"trace_id missing from error payload: {payload!r}"
    )
    trace_id = payload["trace_id"]
    assert isinstance(trace_id, str) and len(trace_id) == 8, (
        f"trace_id has unexpected shape: {trace_id!r}"
    )
    # The error key should also be present (sanity check that we
    # actually triggered the error path).
    assert payload.get("error") == "FILE_DELETED", (
        f"expected error=FILE_DELETED, got payload: {payload!r}"
    )


def test_server_clean_shutdown_on_close_stdin(
    indexed_db_path: Path, tmp_path_factory
):
    """Closing stdin must cause the server to exit within a reasonable
    timeout. A regression that holds the process open after stdin EOF
    would hang every MCP client at teardown.

    We use a fresh server (not the session fixture) so the shutdown is
    isolated from other tests that still need the long-lived server.
    """
    log_dir = tmp_path_factory.mktemp("ios-graphrag-mcp-integration-shutdown-logs")
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(indexed_db_path)
    env["GRAPHRAG_LOG_DIR"] = str(log_dir)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ios_graphrag.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    try:
        _wait_for_ready(proc)
        # Cleanly close stdin — this is the MCP-spec shutdown signal.
        assert proc.stdin is not None
        proc.stdin.close()
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail("server did not exit within 10s of stdin close")
        # Don't assert on the exact exit code: anyio cancellation paths
        # in the mcp.run() teardown can surface as non-zero even on a
        # successful close. The contract is "exits in a reasonable
        # window without hanging".
        assert rc is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
