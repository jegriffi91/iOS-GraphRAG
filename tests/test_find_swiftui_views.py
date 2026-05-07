"""Phase 4.5d -- ``find_swiftui_views`` MCP tool.

Each test loads the indexed fixture DB into the server module's
in-memory state, calls ``server._find_swiftui_views`` directly, and
asserts the returned ``count``/``views`` payload matches expectations
based on the SwiftUIPatterns fixture.

The tests deliberately call the wrapped handler directly (not over an
MCP client transport) so the trace_id decoration is exercised on the
same code path real callers hit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Importing the server module is intentional: we call the registered
# handler directly to bypass the FastMCP transport layer. The decorator
# stack still runs (``_traced_handler`` -> ``_reload_if_stale`` ->
# the SQL query), so behavior matches a real tool invocation.
from ios_graphrag import server


@pytest.fixture(autouse=True)
def _load_fixture_db(indexed_db_path: Path, monkeypatch):
    """Point the server at the session-scoped fixture DB and load it.

    Each test starts from a clean in-memory state (``GRAPH``,
    ``EMBEDDING_MATRIX``, ``_LOADED_MTIME``) so prior calls in the same
    pytest run can't leak state. The ``GRAPH_DB_PATH`` env var is what
    ``_reload_if_stale`` and the SQL handler look at, so we set it here.
    """
    monkeypatch.setenv("GRAPH_DB_PATH", str(indexed_db_path))
    server.load_graph(str(indexed_db_path))
    yield


def test_find_swiftui_views_returns_all_views(indexed_db_path: Path) -> None:
    """No filters -> both LoginView and ProfileBadge appear, with their
    own state_kinds (NOT the union across the file).

    The byte-range scoping in the SQL means LoginView only reports its
    own four wrappers (``state``, ``stateobject``, ``environment``,
    ``appstorage``), and ProfileBadge only reports its own two
    (``binding``, ``observedobject``). ProfileBadge does NOT inherit
    LoginView's wrappers despite sharing a file.
    """
    result = server._find_swiftui_views()
    assert result["count"] == 2, (
        f"expected 2 Views (LoginView + ProfileBadge), got {result['count']}: {result}"
    )
    classes = {v["swift_class"] for v in result["views"]}
    assert classes == {"LoginView", "ProfileBadge"}

    by_class = {v["swift_class"]: v for v in result["views"]}
    login_states = set(by_class["LoginView"]["states"])
    badge_states = set(by_class["ProfileBadge"]["states"])
    # LoginView declares @StateObject, @State, @Environment, @AppStorage.
    assert login_states == {"stateobject", "state", "environment", "appstorage"}, (
        f"LoginView states wrong: {login_states}"
    )
    # ProfileBadge declares @ObservedObject and @Binding only.
    assert badge_states == {"observedobject", "binding"}, (
        f"ProfileBadge states wrong: {badge_states}"
    )

    # trace_id surfaces in the response for log grep-ability.
    assert "trace_id" in result and len(result["trace_id"]) == 8


def test_find_swiftui_views_filter_by_state_kind(indexed_db_path: Path) -> None:
    """``state_kind='stateobject'`` -> only LoginView appears.

    ProfileBadge has no @StateObject (it uses @ObservedObject and
    @Binding), so the byte-range-scoped filter must exclude it.
    """
    result = server._find_swiftui_views(state_kind="stateobject")
    assert result["count"] == 1, (
        f"expected exactly LoginView for state_kind='stateobject', got {result}"
    )
    assert result["views"][0]["swift_class"] == "LoginView"


def test_find_swiftui_views_observable_only(indexed_db_path: Path) -> None:
    """``observable_only=True`` -> Views in files that contain at least
    one @Observable / @Model class.

    AuthState (@Observable) lives in SwiftUIPatterns.swift alongside
    LoginView and ProfileBadge, so both views qualify under the
    file-locality approximation. (A future data-flow analysis could
    narrow this further by tracing each View's declared property
    types -- documented as a known limitation.)
    """
    result = server._find_swiftui_views(observable_only=True)
    classes = {v["swift_class"] for v in result["views"]}
    # SwiftUIPatterns.swift has both AuthState (@Observable) and Recipe
    # is NOT included in the trimmed fixture; only AuthState is. Both
    # Views should qualify because they share the file with AuthState.
    assert classes == {"LoginView", "ProfileBadge"}, (
        f"observable_only should return both Views (file contains @Observable AuthState), "
        f"got {classes}"
    )
    assert result["count"] == 2
