"""
Parametric tests for Swift selector construction.

These exercise ``ios_graphrag.indexer.build_swift_selector`` against the
twelve canonical parameter shapes called out in
``docs/PRODUCTION_ROADMAP.md`` (P1.2). They lock in the contract that
unlabeled parameters (``func foo(_ x: T)``) yield ``foo(_:)`` and never
collapse with the labeled overload ``func foo(x: T)``.

The test parses inline Swift snippets with the same Tree-sitter Swift
grammar the indexer uses, then drives ``build_swift_selector`` directly
against the resulting ``function_declaration`` node. No production code
was reshaped to make this testable; we use the function as-is.

A small set of mirror cases for ``extract_call_selector`` is included at
the bottom for symmetry with the declaration-side coverage.
"""

from __future__ import annotations

import pytest
import tree_sitter_swift
from tree_sitter import Language, Parser

from ios_graphrag.indexer import build_swift_selector, extract_call_selector


@pytest.fixture(scope="module")
def swift_parser() -> Parser:
    """Module-scoped Tree-sitter parser configured for Swift.

    Re-creating the parser per case is unnecessary; parsing is stateless.
    """
    parser = Parser()
    parser.language = Language(tree_sitter_swift.language())
    return parser


def _find_first(node, target_type: str):
    """Depth-first search for the first node of ``target_type``.

    Returns ``None`` if the parse tree has no such node — the caller's
    parametric assertion will surface the failure clearly.
    """
    if node.type == target_type:
        return node
    for child in node.children:
        found = _find_first(child, target_type)
        if found is not None:
            return found
    return None


# Each tuple is (Swift-source-snippet, expected-selector). The snippets
# wrap the function in a class so the grammar produces a clean
# ``function_declaration`` node regardless of top-level vs. method form.
SELECTOR_CASES = [
    pytest.param("func foo() {}", "foo()", id="empty_params"),
    pytest.param("func foo(x: Int) {}", "foo(x:)", id="single_labeled"),
    pytest.param("func foo(_ x: Int) {}", "foo(_:)", id="unlabeled_wildcard"),
    pytest.param(
        "func foo(external internal: Int) {}",
        "foo(external:)",
        id="external_internal_distinct",
    ),
    pytest.param("func foo(x: Int = 5) {}", "foo(x:)", id="default_value"),
    pytest.param("func foo(_ x: inout Int) {}", "foo(_:)", id="inout_unlabeled"),
    pytest.param("func foo(x: Int...) {}", "foo(x:)", id="variadic"),
    pytest.param("func foo<U>(x: U) {}", "foo(x:)", id="generic_clause_ignored"),
    pytest.param(
        "func foo(_ x: Int, y: String) {}",
        "foo(_:y:)",
        id="mixed_unlabeled_labeled",
    ),
    pytest.param(
        "func foo() -> Int { return 0 }",
        "foo()",
        id="return_type_ignored",
    ),
    pytest.param(
        "func foo() async throws -> Int { return 0 }",
        "foo()",
        id="async_throws_ignored",
    ),
    pytest.param(
        "func foo(_ a: Int, _ b: Int) {}",
        "foo(_:_:)",
        id="multi_unlabeled",
    ),
]


@pytest.mark.parametrize("source,expected", SELECTOR_CASES)
def test_build_swift_selector(swift_parser: Parser, source: str, expected: str):
    """build_swift_selector returns the canonical Swift selector for the
    given parameter shape."""
    tree = swift_parser.parse(source.encode("utf-8"))
    func_node = _find_first(tree.root_node, "function_declaration")
    assert func_node is not None, f"No function_declaration parsed from {source!r}"

    selector = build_swift_selector(func_node)
    assert selector == expected, (
        f"build_swift_selector returned {selector!r} for {source!r}; expected {expected!r}"
    )


def test_build_swift_selector_unlabeled_does_not_collapse_with_labeled(
    swift_parser: Parser,
):
    """Regression guard for the original P1.2 bug: ``func test(_ x: T)``
    and ``func test(x: T)`` MUST yield distinct selectors so they don't
    collide as overloads in the index."""
    src_unlabeled = swift_parser.parse(b"func test(_ x: String) {}").root_node
    src_labeled = swift_parser.parse(b"func test(x: String) {}").root_node

    sel_unlabeled = build_swift_selector(_find_first(src_unlabeled, "function_declaration"))
    sel_labeled = build_swift_selector(_find_first(src_labeled, "function_declaration"))

    assert sel_unlabeled == "test(_:)"
    assert sel_labeled == "test(x:)"
    assert sel_unlabeled != sel_labeled, (
        "Selector for unlabeled overload collided with labeled overload — "
        "indexer would record both under the same key"
    )


# --- Mirror cases for extract_call_selector ----------------------------
# These wrap each call expression in a function body so the parser yields
# a clean ``call_expression`` node.

CALL_SELECTOR_CASES = [
    pytest.param('test("hi")', "test(_:)", id="call_unlabeled"),
    pytest.param('test(something: "hi")', "test(something:)", id="call_labeled"),
    pytest.param("foo()", "foo()", id="call_empty"),
    pytest.param(
        "factory.createOperation(for: symbol)",
        "createOperation(for:)",
        id="call_method_with_label",
    ),
    pytest.param("configure(10, 20)", "configure(_:_:)", id="call_two_unlabeled"),
    pytest.param(
        "configure(width: 10, height: 20)",
        "configure(width:height:)",
        id="call_two_labeled",
    ),
]


@pytest.mark.parametrize("call_src,expected", CALL_SELECTOR_CASES)
def test_extract_call_selector(swift_parser: Parser, call_src: str, expected: str):
    """extract_call_selector mirrors build_swift_selector at call sites."""
    wrapped = "func wrapper() { let _ = " + call_src + " }"
    tree = swift_parser.parse(wrapped.encode("utf-8"))
    call_node = _find_first(tree.root_node, "call_expression")
    assert call_node is not None, f"No call_expression parsed from {call_src!r}"

    selector = extract_call_selector(call_node)
    assert selector == expected, (
        f"extract_call_selector returned {selector!r} for {call_src!r}; expected {expected!r}"
    )
