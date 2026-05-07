import argparse
import hashlib
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass

# Force the `requests`-based HF Hub client (vs httpx). Orthogonal to TLS
# verification — kept here because it predates the Phase 2 audit and removing
# it could regress download paths in environments that disable httpx for
# unrelated reasons. Move to _tls.py only if it ever needs to be opt-in.
os.environ.setdefault("HF_HUB_DISABLE_HTTPX", "1")

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tree_sitter_objc
import tree_sitter_swift
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

from . import _migrations, _tls

DB_DEFAULT = "knowledge-graph.sqlite"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 256
MAX_SYMLINK_DEPTH = 10
# Phase 6b moved indexer logs out of the cwd and into
# ``~/Library/Logs/ios-graphrag/indexer.log``. ``LOG_PATH`` is intentionally
# no longer defined here; ``_logging.setup_logging("indexer")`` owns the
# resolved path and returns it for callers that need it.

# Default project-specific model cache. Distinct from the Hugging Face hub
# cache (`~/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5/`)
# which `doctor.py` inspects: that path follows the HF download convention,
# while this one is the staging location populated by `huggingface-cli
# download --local-dir ...` for offline distribution.
DEFAULT_LOCAL_MODEL_DIR = (
    Path.home() / ".cache" / "ios-graphrag" / "models" / "nomic-embed-text-v1.5"
)


def _has_model_weights(model_dir: Path) -> bool:
    """Return True if ``model_dir`` looks like a sentence-transformers checkpoint.

    Sentence-transformers needs ``config.json`` plus one of
    ``model.safetensors`` / ``pytorch_model.bin``. We check for both pieces so
    a partial directory (config without weights, or vice versa) is rejected
    and we fall through to the next candidate in the chain.
    """
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").is_file():
        return False
    return (model_dir / "model.safetensors").is_file() or (
        model_dir / "pytorch_model.bin"
    ).is_file()


def _resolve_model_path() -> str:
    """Resolve the path/identifier passed to ``SentenceTransformer(...)``.

    Resolution order:

    1. ``$IOS_GRAPHRAG_MODEL_DIR`` if set and the directory contains the
       expected weight files.
    2. ``~/.cache/ios-graphrag/models/nomic-embed-text-v1.5/`` if it contains
       the expected weight files.
    3. Fall back to the Hugging Face identifier ``MODEL_NAME``, which means
       ``sentence-transformers`` will use its existing HF hub cache or
       download the model.
    """
    env_dir = os.environ.get("IOS_GRAPHRAG_MODEL_DIR")
    if env_dir:
        env_path = Path(env_dir).expanduser()
        if _has_model_weights(env_path):
            log.info(f"model loaded from IOS_GRAPHRAG_MODEL_DIR: {env_path}")
            return str(env_path)
        log.warning(
            "IOS_GRAPHRAG_MODEL_DIR is set to %s but no model weights "
            "(config.json + model.safetensors|pytorch_model.bin) were found; "
            "falling back to default cache locations.",
            env_path,
        )

    if _has_model_weights(DEFAULT_LOCAL_MODEL_DIR):
        log.info(f"model loaded from local cache: {DEFAULT_LOCAL_MODEL_DIR}")
        return str(DEFAULT_LOCAL_MODEL_DIR)

    log.info(f"using HF identifier {MODEL_NAME!r} (HF hub cache or network download)")
    return MODEL_NAME


BLOCK_DIRS = {"Pods", "Carthage", "DerivedData", ".build", "vendor"}
ALLOWED_EXTS = {".swift", ".h", ".m"}

# Module-level logger. Phase 6b removed the module-level
# ``logging.basicConfig(filename=...)`` that previously installed a
# WARNING-only file handler at import time. The handler chain is now
# configured by :func:`_logging.setup_logging` from ``main()`` (and by the
# embedding worker subprocess on its own re-import). When this module is
# imported as a library and ``setup_logging`` is never called, the standard
# library's "last-resort" handler emits WARNING+ records to stderr — the
# behavior matches the prior import-time semantics for callers that just use
# :func:`log_error`.
log = logging.getLogger(__name__)


# Phase 4b: cached in-process SentenceTransformer singleton.
#
# The previous implementation spawned a fresh ``multiprocessing.spawn``
# subprocess for every ``embed_signatures`` call and reloaded the ~500 MB
# model from disk inside it. The original justification — avoiding
# 'client has been closed' errors from a stale httpx/huggingface_hub
# state and isolating any unconditional SSL bypass — no longer applies:
#
#   * Phase 2 moved all SSL/TLS bypass behind ``_tls.configure_insecure_tls_if_requested``,
#     which is opt-in via ``GRAPHRAG_INSECURE_TLS=1``. Default posture is
#     verify-on, so there's no unconditional state to isolate.
#   * Under the pinned versions (``sentence-transformers>=5.4,<5.5``,
#     ``torch>=2.11,<2.12``, ``huggingface_hub 1.14``, ``httpx 0.28``),
#     loading the model two or three times back-to-back in one process
#     completes cleanly with bit-identical embeddings and no measurable
#     RSS leak. Verified empirically in the Phase 4b audit probe.
#
# Pattern: lazy load on first ``embed_signatures`` call; cache on
# ``_MODEL_CACHE``. ``index_repository`` invocations within the same
# Python process reuse the same model object — important for the
# Phase 4c watcher daemon, which keeps a single process alive across
# many incremental reindexes.
#
# Concurrency: the indexer runs single-threaded over the embed phase
# (``store_embeddings`` is called once per ``index_repository``), so
# no lock is needed. If a future caller threads it, switch to a
# ``threading.Lock`` around the load.
_MODEL_CACHE: Optional["SentenceTransformer"] = None
_MODEL_CACHE_PATH: Optional[str] = None


@dataclass(frozen=True)
class SymbolKey:
    file_path: str
    symbol_name: str
    start_byte: int
    end_byte: int


@dataclass
class Symbol:
    file_path: str
    symbol_name: str
    symbol_type: str
    language: str
    start_byte: int
    end_byte: int
    line_number: Optional[int]
    signature: str
    inherits: List[str]
    conforms: List[str]
    extends: Optional[str]
    calls: List[str]
    selector: Optional[str] = None  # Swift selector: "funcName(_:label:)"
    # Phase 4.5d SwiftUI enrichment columns. NULL for symbols where the
    # detection doesn't apply (e.g. `is_swiftui_view` is only set on
    # struct/class declarations, never on free functions).
    is_swiftui_view: Optional[bool] = None
    is_observable: Optional[bool] = None
    state_kind: Optional[str] = None
    body_kind: Optional[str] = None


# Phase 4.5d: SwiftUI / Combine / SwiftData property-wrapper attribute names
# mapped to the column value the extractor writes into ``nodes.state_kind``.
# Order is documentation-only — when stacked wrappers are encountered (rare
# but possible: e.g. ``@MainActor @State``), the extractor picks the FIRST
# matching attribute it encounters in source order. This is documented in
# README.md's Symbol Types section so the precedence is contractually visible.
_STATE_KIND_BY_ATTRIBUTE: Dict[str, str] = {
    "State": "state",
    "Binding": "binding",
    "StateObject": "stateobject",
    "ObservedObject": "observedobject",
    "EnvironmentObject": "environmentobject",
    "Environment": "environment",
    "AppStorage": "appstorage",
    "SceneStorage": "scenestorage",
    "FetchRequest": "fetchrequest",
    "Query": "query",
    "Published": "published",
}

# Class-level attribute names that flip ``is_observable``. Property-level
# wrappers (e.g. ``@Published``) are intentionally NOT here -- ``is_observable``
# is a class-shape signal, while ``@Published`` is captured per-property via
# ``state_kind``.
_OBSERVABLE_CLASS_ATTRIBUTES = frozenset({"Observable", "Model"})


@dataclass
class ParseResult:
    file_path: str
    symbols: List[Symbol]
    imports: List[str]


def log_error(path: str, message: str, line_number: Optional[int] = None) -> None:
    line_info = line_number if line_number is not None else 0
    logging.warning("%s:%s %s", path, line_info, message)


@contextmanager
def transaction_scope(conn: sqlite3.Connection, use_savepoint: bool, name: str):
    if use_savepoint:
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield
        except Exception:
            conn.execute(f"ROLLBACK TO {name}")
            conn.execute(f"RELEASE {name}")
            raise
        else:
            conn.execute(f"RELEASE {name}")
    else:
        with conn:
            yield


def should_index(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() not in ALLOWED_EXTS:
        return False
    if any(part in BLOCK_DIRS for part in p.parts):
        return False
    if any(part.endswith("Tests") for part in p.parts):
        return False
    name = p.name
    if name.endswith(".generated.swift"):
        return False
    if name.endswith("Tests.swift"):
        return False
    if p.suffix.lower() == ".swift" and "Mock" in name:
        return False
    return True


def is_blocked_dir(name: str) -> bool:
    return name in BLOCK_DIRS or name.endswith("Tests")


def collect_indexable_files(repo_root: str) -> List[str]:
    root = Path(repo_root).resolve()
    files: List[str] = []
    seen_files: set[str] = set()
    seen_dirs: set[Path] = set()
    stack: List[Tuple[Path, int]] = [(root, 0)]

    while stack:
        current, depth = stack.pop()
        try:
            real_current = current.resolve()
        except FileNotFoundError:
            log_error(str(current), "Broken symlink")
            continue
        if real_current in seen_dirs:
            continue
        seen_dirs.add(real_current)
        try:
            for entry in os.scandir(real_current):
                entry_path = Path(entry.path)
                if entry.is_symlink():
                    target_path = Path(os.path.realpath(entry.path))
                    if not target_path.exists():
                        log_error(entry.path, "Broken symlink")
                        continue
                    if target_path.is_dir():
                        if depth + 1 > MAX_SYMLINK_DEPTH:
                            log_error(entry.path, "Symlink depth exceeded")
                            continue
                        if root not in target_path.parents and target_path != root:
                            continue
                        if is_blocked_dir(target_path.name):
                            continue
                        stack.append((target_path, depth + 1))
                        continue
                    if target_path.is_file():
                        canonical = target_path.resolve()
                        if root not in canonical.parents and canonical != root:
                            continue
                        if should_index(str(canonical)) and str(canonical) not in seen_files:
                            files.append(str(canonical))
                            seen_files.add(str(canonical))
                        continue
                if entry.is_dir(follow_symlinks=False):
                    if is_blocked_dir(entry_path.name):
                        continue
                    stack.append((entry_path, depth))
                elif entry.is_file(follow_symlinks=False):
                    canonical = Path(os.path.realpath(entry.path))
                    if root not in canonical.parents and canonical != root:
                        continue
                    if should_index(str(canonical)) and str(canonical) not in seen_files:
                        files.append(str(canonical))
                        seen_files.add(str(canonical))
        except PermissionError as exc:
            log_error(str(real_current), f"Permission error: {exc}")
    return files


def compute_hash(path: str) -> Optional[str]:
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as exc:
        log_error(path, f"Hashing error: {exc}")
        return None


def read_file_text(path: str) -> Tuple[bytes, str, str]:
    data = Path(path).read_bytes()
    try:
        return data, data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data, data.decode("latin-1"), "latin-1"


def extract_signature_line(code_bytes: bytes, start_byte: int) -> str:
    line_end = code_bytes.find(b"\n", start_byte)
    if line_end == -1:
        line_end = len(code_bytes)
    return code_bytes[start_byte:line_end].decode("utf-8", errors="ignore").strip()


def _extract_param_label(param_node) -> Optional[str]:
    """Return the Swift external argument label for a single parameter node.

    Swift parameter forms (and the label they expose to callers):

        x: T                 -> "x"      (external == internal)
        external internal: T -> "external"
        _ name: T            -> "_"      (wildcard / unlabeled call site)

    The Tree-sitter Swift grammar emits a literal ``_`` token for the wildcard
    label. Different tree-sitter-swift versions classify it differently:
    some emit a node whose ``type`` is the literal string ``"_"``; others
    classify it as a ``simple_identifier`` whose text is ``"_"``. We treat
    BOTH as a wildcard regardless of node type so the function does not
    silently regress if the upstream grammar is updated.

    The label-vs-internal-name disambiguation walks the parameter's children
    in order, collecting "identifier-like" tokens (``simple_identifier`` and
    the wildcard ``_``) until we hit the type colon ``:``. Two identifiers
    means external+internal — return the FIRST. One identifier means the
    same name is both external and internal — return it unchanged.
    """
    identifier_tokens: List[Tuple[str, str]] = []  # (kind, text); kind in {"wildcard", "name"}

    for param_child in param_node.children:
        # The colon separates the label/internal-name pair from the type.
        # Anything after ``:`` is type info, modifiers, defaults — not labels.
        if param_child.type == ":":
            break

        text = param_child.text.decode("utf-8", errors="ignore")

        # Wildcard label: accept either a node typed ``"_"`` (some grammar
        # versions) or a ``simple_identifier`` whose text is exactly ``"_"``
        # (other grammar versions).
        if param_child.type == "_" or (param_child.type == "simple_identifier" and text == "_"):
            identifier_tokens.append(("wildcard", "_"))
            continue

        if param_child.type == "simple_identifier":
            identifier_tokens.append(("name", text))

    if not identifier_tokens:
        return None

    # Two identifiers -> "external internal: T"; the FIRST is the call-site label.
    # One identifier -> "x: T"; the only identifier is both label and internal name.
    return identifier_tokens[0][1]


def build_swift_selector(func_node) -> Optional[str]:
    """Build a Swift selector from a function declaration node.

    Swift selectors use the format: ``funcName(_:label:otherLabel:)`` where
    ``_`` represents unlabeled parameters and other labels are preserved.

    Examples::

        func test(_ something: String)         -> "test(_:)"
        func test(something: String)           -> "test(something:)"
        func login(email: String, password: String)
                                               -> "login(email:password:)"
        func foo(external internal: Int)       -> "foo(external:)"

    Generic clauses (``<U>``), return types (``-> T``), and effect clauses
    (``async``, ``throws``) are ignored — they don't appear in the Swift
    selector.
    """
    func_name = None
    param_labels: List[str] = []

    for child in func_node.children:
        if child.type == "simple_identifier" and func_name is None:
            func_name = child.text.decode("utf-8", errors="ignore")
        elif child.type == "parameter":
            label = _extract_param_label(child)
            if label is not None:
                param_labels.append(label + ":")

    if not func_name:
        return None

    if not param_labels:
        return f"{func_name}()"

    return f"{func_name}({''.join(param_labels)})"


def extract_call_selector(call_node) -> Optional[str]:
    """Extract a Swift selector from a call_expression node.

    Parses the call expression AST to build the selector from:
    - Function name from simple_identifier (for direct calls) or
      navigation_expression > navigation_suffix > simple_identifier (for method calls)
    - Argument labels from value_argument_label children, or '_' for unlabeled args

    For example:
        test(something) -> "test(_:)" (unlabeled)
        test(something: value) -> "test(something:)"
        factory.createOperation(for: symbol) -> "createOperation(for:)"
    """
    func_name = None
    arg_labels: List[str] = []

    for child in call_node.children:
        # Get function name - direct call like: test()
        if child.type == "simple_identifier" and func_name is None:
            func_name = child.text.decode("utf-8", errors="ignore")

        # Get function name - method call like: factory.createOperation()
        elif child.type == "navigation_expression" and func_name is None:
            for nav_child in child.children:
                if nav_child.type == "navigation_suffix":
                    for suffix_child in nav_child.children:
                        if suffix_child.type == "simple_identifier":
                            func_name = suffix_child.text.decode("utf-8", errors="ignore")
                            break

        # Get call suffix which contains arguments
        elif child.type == "call_suffix":
            for suffix_child in child.children:
                if suffix_child.type == "value_arguments":
                    for arg_child in suffix_child.children:
                        if arg_child.type == "value_argument":
                            # Check if this argument has a label
                            has_label = False
                            for va_child in arg_child.children:
                                if va_child.type == "value_argument_label":
                                    # Get the label text
                                    for label_child in va_child.children:
                                        if label_child.type == "simple_identifier":
                                            arg_labels.append(
                                                label_child.text.decode("utf-8", errors="ignore")
                                                + ":"
                                            )
                                            has_label = True
                                            break
                            if not has_label:
                                # Unlabeled argument uses _
                                arg_labels.append("_:")

    if not func_name:
        return None

    if not arg_labels:
        return f"{func_name}()"

    return f"{func_name}({''.join(arg_labels)})"


def normalize_type_name(name: str) -> str:
    return name.split("<", 1)[0].strip()


def extract_inheritance_types(node, code_bytes: bytes) -> List[str]:
    """Extract inherited types from a declaration node.

    Compatible with tree-sitter-swift grammars that use either:
    - inheritance_clause > type_identifier (older grammars)
    - inheritance_specifier > user_type > type_identifier (newer grammars)
    """
    types: List[str] = []
    for child in node.children:
        # Handle newer grammar: inheritance_specifier
        if child.type == "inheritance_specifier":
            # Look for user_type > type_identifier
            for sub in child.children:
                if sub.type == "user_type":
                    for inner in sub.children:
                        if inner.type == "type_identifier":
                            types.append(
                                normalize_type_name(inner.text.decode("utf-8", errors="ignore"))
                            )
                elif sub.type == "type_identifier":
                    types.append(normalize_type_name(sub.text.decode("utf-8", errors="ignore")))
        # Handle older grammar: inheritance_clause
        elif child.type == "inheritance_clause":
            for sub in child.children:
                if sub.type in {"type_identifier", "identifier", "generic_type"}:
                    types.append(normalize_type_name(sub.text.decode("utf-8", errors="ignore")))
                elif sub.type == "scoped_type_identifier":
                    types.append(normalize_type_name(sub.text.decode("utf-8", errors="ignore")))
    return [t for t in types if t]


def extract_property_name(prop_node) -> Optional[str]:
    """Return the bound name from a Swift ``property_declaration`` node.

    The Tree-sitter Swift grammar nests the bound identifier inside a
    ``pattern`` child::

        property_declaration
          value_binding_pattern (var | let)
          pattern
            simple_identifier   <- the name we want
          type_annotation
          [computed_property | willset_didset_block | initializer]

    Tuple bindings (``let (x, y) = ...``) produce a ``pattern`` whose
    children are themselves nested ``pattern`` nodes; we don't try to
    decompose those — we return ``None`` and the caller skips the
    declaration. Protocol property requirements use a different node
    type (``protocol_property_declaration``), which this function is
    NOT called on.
    """
    for child in prop_node.children:
        if child.type != "pattern":
            continue
        # Look for a single simple_identifier directly under pattern.
        ident_children = [c for c in child.children if c.type == "simple_identifier"]
        if len(ident_children) == 1 and not any(c.type == "pattern" for c in child.children):
            return ident_children[0].text.decode("utf-8", errors="ignore")
        # Tuple binding (multiple nested patterns) -- not supported here.
        return None
    return None


def is_computed_property(prop_node) -> bool:
    """Return True if the ``property_declaration`` has a getter/setter block.

    Tree-sitter Swift represents both the simple-body form
    (``var foo: Int { return 5 }``) and the explicit get/set form
    (``var foo: Int { get { ... } set { ... } }``) with a child node of
    type ``computed_property``. The observer-only form
    (``var foo: T = init { didSet { ... } }``) uses
    ``willset_didset_block`` and is treated as a stored property.
    """
    return any(child.type == "computed_property" for child in prop_node.children)


def _extract_enum_case_param_labels(enum_type_parameters_node) -> List[str]:
    """Extract argument labels from an ``enum_type_parameters`` node.

    Unlike ``function_declaration`` whose parameters are wrapped in
    ``parameter`` nodes, the Tree-sitter Swift grammar lays the children
    of ``enum_type_parameters`` out FLAT, separated by ``,`` tokens::

        enum_type_parameters
          (
          simple_identifier   <- label "name"
          :
          user_type
          ,
          simple_identifier   <- label "fn"
          :
          function_type
          )

    Unlabeled associated values omit the leading ``simple_identifier``
    (the type starts directly after ``(`` or ``,``). A leading wildcard
    (``case mixed(_ first: String)``) appears as a ``wildcard_pattern``
    or ``simple_identifier`` whose text is ``"_"``.

    Returns labels in source order, with ``"_"`` for unlabeled segments.
    """
    labels: List[str] = []
    # Split children into per-comma segments (skipping the outer parens).
    segments: List[List[Tuple[str, str]]] = []
    current: List[Tuple[str, str]] = []
    for child in enum_type_parameters_node.children:
        if child.type in {"(", ")"}:
            continue
        if child.type == ",":
            if current:
                segments.append(current)
            current = []
            continue
        text = child.text.decode("utf-8", errors="ignore")
        current.append((child.type, text))
    if current:
        segments.append(current)

    for segment in segments:
        # Walk segment until we hit ":" — anything before is label/internal,
        # anything after is type info we don't need for the selector.
        label: Optional[str] = None
        seen_identifier = False
        for kind, text in segment:
            if kind == ":":
                break
            if kind == "wildcard_pattern" or (kind == "simple_identifier" and text == "_"):
                if not seen_identifier:
                    label = "_"
                    seen_identifier = True
                continue
            if kind == "simple_identifier":
                if not seen_identifier:
                    label = text
                    seen_identifier = True
        if label is None:
            # No leading identifier in this segment -> the type is positional,
            # which Swift renders as an unlabeled associated value.
            labels.append("_")
        else:
            labels.append(label)
    return labels


def _extract_enum_case_names(enum_entry_node) -> List[Tuple[str, int, int]]:
    """Return ``[(case_name, start_byte, end_byte)]`` for each case in an
    ``enum_entry``.

    The Tree-sitter Swift grammar represents
    ``case red, green, blue`` as a single ``enum_entry`` node containing
    multiple ``simple_identifier`` children separated by ``,``. We emit
    one tuple per identifier so the caller can issue one row per case.

    For cases with associated values
    (``case custom(name: String, fn: ...)``) or raw values
    (``case loading = "loading"``), only one ``simple_identifier`` directly
    follows the ``case`` keyword (the rest of the children are
    ``enum_type_parameters`` or the raw-value expression).
    """
    pairs: List[Tuple[str, int, int]] = []
    seen_punct_or_param = False
    for child in enum_entry_node.children:
        # The case body shape is:
        #   case <name>[(...)] [, <name>[(...)]] [= <expr>]
        # If we hit enum_type_parameters or a raw-value '=', subsequent
        # simple_identifiers are NOT case names.
        if child.type in {"enum_type_parameters", "="}:
            seen_punct_or_param = True
            continue
        if child.type == "simple_identifier" and not seen_punct_or_param:
            text = child.text.decode("utf-8", errors="ignore")
            pairs.append((text, child.start_byte, child.end_byte))
        # Note: a `,` separator does NOT flip seen_punct_or_param, because
        # `case red, green, blue` continues to enumerate names.
    return pairs


def _extract_enum_case_selector(enum_entry_node, case_name: str) -> Optional[str]:
    """Build a Swift call-site selector for a case with associated values.

    Returns ``"custom(name:fn:)"`` for ``case custom(name: String, fn: ...)``,
    ``"unlabeled(_:)"`` for ``case unlabeled(String)``, or ``None`` if the
    case has no associated values (the caller leaves selector NULL on
    those rows).
    """
    for child in enum_entry_node.children:
        if child.type != "enum_type_parameters":
            continue
        labels = _extract_enum_case_param_labels(child)
        if not labels:
            return f"{case_name}()"
        return f"{case_name}({''.join(label + ':' for label in labels)})"
    return None


def _attribute_names(node) -> List[str]:
    """Return the type-identifier names of every ``@Foo(...)`` attribute on ``node``.

    The Tree-sitter Swift grammar nests attributes under a ``modifiers``
    child, where each ``attribute`` node contains an ``@`` token followed by
    a ``user_type`` whose ``type_identifier`` is the attribute's bare name::

        modifiers
          attribute
            @
            user_type
              type_identifier   <- e.g. "State", "Observable"
            ( ... )             <- optional argument list

    Returns names in source order. An empty list means the node has no
    attached attributes. Used for both class-level (``@Observable``,
    ``@Model``) and property-level (``@State``, ``@Binding``, etc.) lookups.
    """
    names: List[str] = []
    for child in node.children:
        if child.type != "modifiers":
            continue
        for modifier_child in child.children:
            if modifier_child.type != "attribute":
                continue
            for attr_child in modifier_child.children:
                if attr_child.type == "user_type":
                    for inner in attr_child.children:
                        if inner.type == "type_identifier":
                            names.append(inner.text.decode("utf-8", errors="ignore"))
                            break
                    break
    return names


def _detect_swiftui_view(class_decl_node, actual_type: str) -> bool:
    """Return True if ``class_decl_node`` declares a SwiftUI View-related type.

    Triggers: any inheritance specifier whose type identifier is exactly
    ``View``, exactly ``ViewModifier``, or ends with ``View`` (e.g.
    ``CustomView``). Suffix matching is intentional -- SwiftUI types in the
    real codebase routinely subclass app-specific intermediate Views, and
    the column is meant as a "this participates in the SwiftUI view tree"
    flag rather than strict-View-conformance.

    Only fires for ``struct`` and ``class`` (Swift's two View-eligible
    type kinds); ``enum``/``protocol``/``extension``/``actor`` cannot
    declare a View, so we skip them to keep the column NULL there.
    """
    if actual_type not in {"struct", "class"}:
        return False
    inheritance_types = []
    for child in class_decl_node.children:
        if child.type == "inheritance_specifier":
            for sub in child.children:
                if sub.type == "user_type":
                    for inner in sub.children:
                        if inner.type == "type_identifier":
                            inheritance_types.append(
                                inner.text.decode("utf-8", errors="ignore")
                            )
                elif sub.type == "type_identifier":
                    inheritance_types.append(sub.text.decode("utf-8", errors="ignore"))
    for type_name in inheritance_types:
        normalized = normalize_type_name(type_name)
        if normalized == "View" or normalized == "ViewModifier" or normalized.endswith("View"):
            return True
    return False


def _detect_observable_class(class_decl_node, actual_type: str) -> bool:
    """Return True if ``class_decl_node`` is a class with @Observable or @Model.

    Per Phase 4.5d's policy decision: ``is_observable`` reflects ONLY the
    class-level macro/attribute (``@Observable`` from Swift Observation,
    ``@Model`` from SwiftData). ``@Published`` is intentionally NOT
    considered here -- it's a property-level Combine wrapper captured via
    ``state_kind`` instead.

    Only fires for ``class``; structs/enums/protocols cannot be
    @Observable/@Model in the language.
    """
    if actual_type != "class":
        return False
    return any(name in _OBSERVABLE_CLASS_ATTRIBUTES for name in _attribute_names(class_decl_node))


def _detect_property_state_kind(prop_node) -> Optional[str]:
    """Return the SwiftUI/Combine state-kind for a property, or None.

    Iterates the property's attributes in source order and returns the
    value mapped to the FIRST recognized wrapper. Stacked wrappers
    (``@MainActor @State`` etc.) are valid Swift, but we deliberately
    only surface the first matching one because:
      (a) the wrappers we care about are mutually exclusive in practice,
      (b) the column is a flat TEXT, not a list, so picking-and-documenting
          is simpler than encoding a precedence rule per pair.
    """
    for name in _attribute_names(prop_node):
        kind = _STATE_KIND_BY_ATTRIBUTE.get(name)
        if kind is not None:
            return kind
    return None


def _has_some_view_return(prop_node) -> bool:
    """Return True if the property's type annotation is ``some View``.

    Tree-sitter Swift represents ``var body: some View { ... }`` as::

        property_declaration
          ...
          type_annotation
            :
            opaque_type
              some
              user_type
                type_identifier   <- "View"

    We walk to the ``opaque_type`` child of the ``type_annotation`` and
    inspect the inner ``user_type``'s ``type_identifier``. Anything that
    isn't exactly ``View`` (``some Hashable``, ``some Collection``,
    ``some MyView``) is intentionally NOT marked as ``viewbody`` -- the
    column is reserved for the View-tree primary entry point, which by
    convention is ``some View``.
    """
    for child in prop_node.children:
        if child.type != "type_annotation":
            continue
        for sub in child.children:
            if sub.type != "opaque_type":
                continue
            for inner in sub.children:
                if inner.type == "user_type":
                    for tid in inner.children:
                        if tid.type == "type_identifier":
                            return tid.text.decode("utf-8", errors="ignore") == "View"
    return False


def _detect_property_body_kind(prop_node, prop_name: Optional[str]) -> Optional[str]:
    """Return ``body_kind`` for a computed-property declaration, or None.

    Two paths:
      * ``var body: some View { ... }`` -> ``"viewbody"``. The combination
        of name == "body" AND opaque-type return ``some View`` is what
        SwiftUI's protocol requires; matching both reduces false positives
        from accidental ``body``-named properties on non-View types.
      * ``@ViewBuilder var foo: some View { ... }`` -> ``"resultbuilder"``.
        Any computed property tagged with ``@ViewBuilder`` -- distinct from
        the SwiftUI `body` entry point.

    Returns None for stored properties and for computed properties that
    match neither pattern.
    """
    if not is_computed_property(prop_node):
        return None
    attrs = _attribute_names(prop_node)
    if "ViewBuilder" in attrs:
        return "resultbuilder"
    if prop_name == "body" and _has_some_view_return(prop_node):
        return "viewbody"
    return None


def _function_has_some_view_return(func_node) -> bool:
    """Return True if the function declaration has return type ``some View``.

    Tree-sitter Swift puts the return type as a top-level
    ``opaque_type`` child of ``function_declaration`` (no intervening
    ``type_annotation`` wrapper -- functions use ``->`` syntax).
    """
    for child in func_node.children:
        if child.type != "opaque_type":
            continue
        for inner in child.children:
            if inner.type == "user_type":
                for tid in inner.children:
                    if tid.type == "type_identifier":
                        return tid.text.decode("utf-8", errors="ignore") == "View"
    return False


def _detect_function_body_kind(func_node) -> Optional[str]:
    """Return ``body_kind`` for a function declaration, or None.

    A function annotated with ``@ViewBuilder`` (or any other Swift result
    builder; we currently only flag ViewBuilder) is tagged
    ``"resultbuilder"``. Free functions whose return type is ``some View``
    but lack the ``@ViewBuilder`` annotation are still 'just functions'
    in the SwiftUI sense -- the result-builder transform doesn't apply,
    so we leave ``body_kind`` NULL for them.

    Functions that match neither pattern get NULL (rather than
    ``"regular"``) so the column stays sparse and queries against
    ``WHERE body_kind IS NOT NULL`` only surface the symbols that
    participate in the View tree.
    """
    if "ViewBuilder" in _attribute_names(func_node):
        return "resultbuilder"
    return None


def extract_swift_symbols(code_bytes: bytes, tree) -> Tuple[List[Symbol], List[str]]:
    """Extract symbols from Swift code using manual tree walking.

    This approach is more robust across different tree-sitter versions than using queries.
    """
    symbols: List[Symbol] = []
    imports: List[str] = []

    def walk_node(node):
        """Recursively walk the tree and extract symbols."""
        node_type = node.type

        # Handle imports
        if node_type == "import_declaration":
            for child in node.children:
                if child.type in {"identifier", "type_identifier"}:
                    imports.append(child.text.decode("utf-8", errors="ignore").split(".")[0])
                    break
            return  # Don't recurse into imports

        # Handle class_declaration (which covers class, struct, enum, protocol, extension, actor)
        if node_type == "class_declaration":
            actual_type = None
            name_text = None

            for child in node.children:
                child_type = child.type
                # Check for type keywords
                if child_type in {"struct", "class", "enum", "protocol", "extension", "actor"}:
                    actual_type = child_type
                # Look for the name (type_identifier or user_type)
                elif child_type == "type_identifier":
                    name_text = child.text.decode("utf-8", errors="ignore")
                elif child_type == "user_type" and name_text is None:
                    # For extensions: extension String { }
                    for inner in child.children:
                        if inner.type == "type_identifier":
                            name_text = inner.text.decode("utf-8", errors="ignore")
                            break

            if actual_type and name_text:
                signature = extract_signature_line(code_bytes, node.start_byte)
                line_number = node.start_point[0] + 1
                inherits = []
                conforms = []
                extends = None

                # Extract inheritance/conformance
                inheritance_types = extract_inheritance_types(node, code_bytes)
                if actual_type == "class" and inheritance_types:
                    inherits = [inheritance_types[0]]
                    conforms = inheritance_types[1:]
                elif actual_type in {"struct", "enum", "protocol", "actor"}:
                    conforms = inheritance_types
                elif actual_type == "extension":
                    extends = name_text
                    conforms = inheritance_types

                # Phase 4.5d SwiftUI enrichment.
                is_swiftui_view = _detect_swiftui_view(node, actual_type) or None
                is_observable = _detect_observable_class(node, actual_type) or None

                symbols.append(
                    Symbol(
                        file_path="",
                        symbol_name=name_text,
                        symbol_type=actual_type,
                        language="swift",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        line_number=line_number,
                        signature=signature,
                        inherits=inherits,
                        conforms=conforms,
                        extends=extends,
                        calls=[],
                        is_swiftui_view=is_swiftui_view,
                        is_observable=is_observable,
                    )
                )
            # Still recurse to find nested types and (for enums) enum cases.
            # The grammar uses ``enum_class_body`` for enums and
            # ``class_body`` for class/struct/protocol/extension/actor; both
            # may contain nested types or member declarations the walker
            # needs to descend into.
            for child in node.children:
                if child.type in {"class_body", "enum_class_body"}:
                    for inner in child.children:
                        walk_node(inner)
            return

        # Handle protocol_declaration (tree-sitter-swift uses a separate node type for protocols)
        if node_type == "protocol_declaration":
            name_text = None
            for child in node.children:
                if child.type == "type_identifier":
                    name_text = child.text.decode("utf-8", errors="ignore")
                    break

            if name_text:
                signature = extract_signature_line(code_bytes, node.start_byte)
                line_number = node.start_point[0] + 1

                # Extract conformance (protocols can inherit from other protocols)
                conforms = extract_inheritance_types(node, code_bytes)

                symbols.append(
                    Symbol(
                        file_path="",
                        symbol_name=name_text,
                        symbol_type="protocol",
                        language="swift",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        line_number=line_number,
                        signature=signature,
                        inherits=[],
                        conforms=conforms,
                        extends=None,
                        calls=[],
                    )
                )
            return

        # Handle function_declaration
        if node_type == "function_declaration":
            name_text = None
            for child in node.children:
                if child.type == "simple_identifier":
                    name_text = child.text.decode("utf-8", errors="ignore")
                    break

            if name_text:
                signature = extract_signature_line(code_bytes, node.start_byte)
                line_number = node.start_point[0] + 1
                selector = build_swift_selector(node)

                # Extract function calls using tree-sitter to get full selectors
                calls: List[str] = []

                def extract_calls_from_node(n):
                    """Recursively find call_expression nodes and extract their selectors."""
                    if n.type == "call_expression":
                        call_selector = extract_call_selector(n)
                        if call_selector and not call_selector.startswith(name_text + "("):
                            # Avoid recording self-calls
                            calls.append(call_selector)
                    for child in n.children:
                        extract_calls_from_node(child)

                # Walk the function body to find all calls
                for child in node.children:
                    if child.type == "function_body":
                        extract_calls_from_node(child)

                # Phase 4.5d SwiftUI enrichment for functions: only @ViewBuilder
                # functions get a body_kind. Plain `some View`-returning functions
                # without @ViewBuilder do not get the result-builder transform, so
                # leaving body_kind NULL there is intentional.
                body_kind = _detect_function_body_kind(node)

                symbols.append(
                    Symbol(
                        file_path="",
                        symbol_name=name_text,
                        symbol_type="function",
                        language="swift",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        line_number=line_number,
                        signature=signature,
                        inherits=[],
                        conforms=[],
                        extends=None,
                        calls=calls,
                        selector=selector,
                        body_kind=body_kind,
                    )
                )
            return

        # Handle property_declaration (Phase 5 P0). Captures stored, computed,
        # observed, lazy, and static properties. Distinguishes the
        # 'computed_property' kind via is_computed_property() (presence of a
        # getter/setter block); everything else is 'property'. Tuple bindings
        # like ``let (x, y) = ...`` return None from extract_property_name and
        # are silently skipped (rare in iOS Swift, complex to map 1-to-1).
        if node_type == "property_declaration":
            prop_name = extract_property_name(node)
            if prop_name:
                signature = extract_signature_line(code_bytes, node.start_byte)
                line_number = node.start_point[0] + 1
                prop_kind = "computed_property" if is_computed_property(node) else "property"
                # Phase 4.5d SwiftUI enrichment for properties: state_kind from
                # @State/@Binding/@StateObject/etc., body_kind from `body: some
                # View` or @ViewBuilder.
                state_kind = _detect_property_state_kind(node)
                body_kind = _detect_property_body_kind(node, prop_name)
                symbols.append(
                    Symbol(
                        file_path="",
                        symbol_name=prop_name,
                        symbol_type=prop_kind,
                        language="swift",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        line_number=line_number,
                        signature=signature,
                        inherits=[],
                        conforms=[],
                        extends=None,
                        calls=[],
                        state_kind=state_kind,
                        body_kind=body_kind,
                    )
                )
            return

        # Handle init_declaration (Phase 5 P0). Covers plain init, failable
        # init?, throwing init, and modifier-tagged inits (convenience,
        # required, override). Symbol name is the literal "init"; the parent
        # type plus selector disambiguate overloads. Selector reuses the
        # existing build_swift_selector machinery -- it walks "simple_identifier"
        # for the function name, but tree-sitter-swift emits "init" as a
        # keyword token, so we override the name to "init" while still
        # collecting parameter labels via _extract_param_label.
        if node_type == "init_declaration":
            line_number = node.start_point[0] + 1
            signature = extract_signature_line(code_bytes, node.start_byte)
            param_labels: List[str] = []
            for child in node.children:
                if child.type == "parameter":
                    label = _extract_param_label(child)
                    if label is not None:
                        param_labels.append(label + ":")
            if param_labels:
                selector = f"init({''.join(param_labels)})"
            else:
                selector = "init()"
            symbols.append(
                Symbol(
                    file_path="",
                    symbol_name="init",
                    symbol_type="initializer",
                    language="swift",
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    line_number=line_number,
                    signature=signature,
                    inherits=[],
                    conforms=[],
                    extends=None,
                    calls=[],
                    selector=selector,
                )
            )
            return

        # Handle deinit_declaration (Phase 5 P0). Deinitializers take no
        # parameters and are unique within their declaring type, so the
        # symbol name "deinit" plus file_path + parent type uniquely
        # identify the row. Selector is left NULL because deinit has no
        # call-site form (it's invoked implicitly by ARC).
        if node_type == "deinit_declaration":
            line_number = node.start_point[0] + 1
            signature = extract_signature_line(code_bytes, node.start_byte)
            symbols.append(
                Symbol(
                    file_path="",
                    symbol_name="deinit",
                    symbol_type="deinitializer",
                    language="swift",
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    line_number=line_number,
                    signature=signature,
                    inherits=[],
                    conforms=[],
                    extends=None,
                    calls=[],
                )
            )
            return

        # Handle subscript_declaration (Phase 5 P2). Symbol name is the
        # literal "subscript" -- overloads are disambiguated by the
        # selector, which mirrors function selectors via
        # _extract_param_label. Examples:
        #     subscript(key: String) -> Double?         -> "subscript(key:)"
        #     subscript(index: Int, withDefault f: Double) -> Double
        #                                                -> "subscript(index:withDefault:)"
        # Tree-sitter-swift exposes parameters as direct ``parameter``
        # children, so the same builder used for ``init`` works here.
        if node_type == "subscript_declaration":
            line_number = node.start_point[0] + 1
            signature = extract_signature_line(code_bytes, node.start_byte)
            param_labels: List[str] = []
            for child in node.children:
                if child.type == "parameter":
                    label = _extract_param_label(child)
                    if label is not None:
                        param_labels.append(label + ":")
            selector = f"subscript({''.join(param_labels)})"
            symbols.append(
                Symbol(
                    file_path="",
                    symbol_name="subscript",
                    symbol_type="subscript",
                    language="swift",
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    line_number=line_number,
                    signature=signature,
                    inherits=[],
                    conforms=[],
                    extends=None,
                    calls=[],
                    selector=selector,
                )
            )
            return

        # Handle typealias_declaration (Phase 5 P2). The first
        # ``type_identifier`` child is the alias name. Generic
        # typealiases (``Result<T> = Swift.Result<T, Error>``) keep the
        # bare name -- the type_identifier covers ``Result`` and the
        # subsequent ``type_parameters`` is a separate sibling we do not
        # walk into. selector stays NULL because typealiases are not
        # invocable.
        if node_type == "typealias_declaration":
            alias_name: Optional[str] = None
            for child in node.children:
                if child.type == "type_identifier":
                    alias_name = child.text.decode("utf-8", errors="ignore")
                    break
            if alias_name:
                line_number = node.start_point[0] + 1
                signature = extract_signature_line(code_bytes, node.start_byte)
                symbols.append(
                    Symbol(
                        file_path="",
                        symbol_name=alias_name,
                        symbol_type="typealias",
                        language="swift",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        line_number=line_number,
                        signature=signature,
                        inherits=[],
                        conforms=[],
                        extends=None,
                        calls=[],
                    )
                )
            return

        # Handle enum_entry (Phase 5 P1). One ``enum_entry`` node may declare
        # multiple cases on a single line (``case red, green, blue``); we
        # emit one Symbol per case name. For cases with associated values
        # we build a call-site selector via _extract_enum_case_selector
        # (e.g. ``custom(name:fn:)``); raw-value enums (``case idle =
        # "idle"``) and bare cases get selector=NULL because they have no
        # invocation form. The line_number tracks the case identifier's
        # actual start so multi-case-per-line entries differ even when
        # they share the same enum_entry start.
        if node_type == "enum_entry":
            entry_signature = extract_signature_line(code_bytes, node.start_byte)
            for case_name, case_start, case_end in _extract_enum_case_names(node):
                line_number = node.start_point[0] + 1
                # If the entry has associated values, we keep the selector;
                # otherwise selector stays NULL (matches Phase 5 P0 deinit).
                # Multi-case-per-line shares the same enum_type_parameters,
                # but Swift forbids associated values on multi-case lines,
                # so this is unambiguous in practice.
                selector = _extract_enum_case_selector(node, case_name)
                symbols.append(
                    Symbol(
                        file_path="",
                        symbol_name=case_name,
                        symbol_type="enum_case",
                        language="swift",
                        start_byte=case_start,
                        end_byte=case_end,
                        line_number=line_number,
                        signature=entry_signature,
                        inherits=[],
                        conforms=[],
                        extends=None,
                        calls=[],
                        selector=selector,
                    )
                )
            return

        # Recurse into children
        for child in node.children:
            walk_node(child)

    # Start walking from root
    walk_node(tree.root_node)

    return symbols, sorted(set(imports))


def _extract_objc_method_selector(method_node) -> Optional[str]:
    """Build the canonical ObjC selector form for a method declaration/definition.

    Tree-sitter-objc emits method nodes as a sequence of identifiers
    interleaved with ``method_parameter`` children. The grammar shape is::

        method_declaration / method_definition
            - / +                        (instance vs class method)
            method_type                  (return type, ignored)
            identifier                   (selector keyword 1, e.g. "addValue")
            method_parameter             (":(type)name" -- only present if
                                          there is a colon-terminated keyword)
            identifier                   (selector keyword 2, e.g. "toValue")
            method_parameter
            ...

    Examples::

        - (void)foo                        -> "foo"           (nullary)
        - (double)addValue:(double)a       -> "addValue:"     (one keyword)
        - (double)addValue:(double)a toValue:(double)b
                                           -> "addValue:toValue:"

    Returns ``None`` if no identifier child exists (defensive; should not
    happen in well-formed source).
    """
    keyword_parts: List[str] = []
    has_method_parameter = False

    for child in method_node.children:
        if child.type == "identifier":
            # Every direct ``identifier`` child belongs to the selector.
            # The grammar interleaves ``identifier`` and
            # ``method_parameter`` so child order corresponds to
            # selector keyword order.
            keyword_parts.append(child.text.decode("utf-8", errors="ignore"))
        elif child.type == "method_parameter":
            has_method_parameter = True

    if not keyword_parts:
        return None

    if not has_method_parameter:
        # Nullary method like ``- (void)foo`` -- selector is just the
        # identifier with no trailing colon.
        return keyword_parts[0]

    # Keyword method: every keyword gets a trailing colon.
    return "".join(part + ":" for part in keyword_parts)


def _extract_objc_property_name(property_node) -> Optional[str]:
    """Return the declared name from an ObjC ``property_declaration`` node.

    Tree-sitter-objc nests the name under
    ``property_declaration > struct_declaration > struct_declarator``.
    The declarator may be wrapped in a ``pointer_declarator`` (for
    object-typed properties like ``NSString *name``) or be a bare
    ``identifier`` (for primitive scalar properties like ``NSInteger
    precision``). Both shapes resolve to the inner ``identifier`` whose
    field name on its parent is ``declarator``.
    """
    for child in property_node.children:
        if child.type != "struct_declaration":
            continue
        for sub in child.children:
            if sub.type != "struct_declarator":
                continue
            # Dig through optional pointer_declarator.
            for inner in sub.children:
                if inner.type == "identifier":
                    return inner.text.decode("utf-8", errors="ignore")
                if inner.type == "pointer_declarator":
                    for deep in inner.children:
                        if deep.type == "identifier":
                            return deep.text.decode("utf-8", errors="ignore")
    return None


def extract_objc_symbols(code_bytes: bytes, tree) -> Tuple[List[Symbol], List[str]]:
    """Extract symbols from Objective-C source via manual tree walking.

    Mirrors ``extract_swift_symbols``: walks the tree-sitter-objc AST and
    emits one ``Symbol`` per ``@interface``, ``@protocol``,
    ``@implementation``, ``@property``, ``- / + method`` declaration. The
    grammar (tree-sitter-objc 3.0.x) uses these node types:

        class_interface       -> @interface block (and categories: when a
                                 ``category`` field is present, emit a
                                 ``category`` symbol for the parens name
                                 instead of an ``objc_class``).
        class_implementation  -> @implementation block (suppressed --
                                 already emitted as objc_class via the
                                 header; emitting again would double-count).
        protocol_declaration  -> @protocol block.
        property_declaration  -> @property line.
        method_declaration    -> ``- / +`` method in a header.
        method_definition     -> ``- / +`` method in an .m file (wrapped
                                 in implementation_definition).

    Methods get a canonical ObjC selector (``addValue:toValue:``) stored
    in ``selector``. Categories carry the parenthesized name as their
    symbol name (e.g. ``Trig``); the @category-on-@implementation form
    follows the same rule for symmetry but is not emitted as an
    ``objc_class`` -- it shares a row with the header declaration.
    """
    symbols: List[Symbol] = []
    imports: List[str] = []
    # Track which (class_name, category_name) categories we've already
    # emitted so a category that exists in BOTH the header and the .m
    # only produces one row in the index.
    seen_categories: set = set()
    # Track classes we've already emitted so a header + implementation
    # pair produces one objc_class row, not two.
    seen_classes: set = set()

    def emit_methods(parent_node, parent_signature_line: int) -> None:
        """Walk a class/protocol body and emit method/property symbols.

        Methods may be wrapped in ``implementation_definition`` (for .m
        files) or appear as bare ``method_declaration`` children (for
        .h files). Properties are always direct children.
        """
        for child in parent_node.children:
            method_node = None
            if child.type == "method_declaration":
                method_node = child
            elif child.type == "implementation_definition":
                # Implementation defs wrap the actual method_definition.
                for inner in child.children:
                    if inner.type == "method_definition":
                        method_node = inner
                        break
            elif child.type == "method_definition":
                method_node = child
            elif child.type == "property_declaration":
                prop_name = _extract_objc_property_name(child)
                if prop_name:
                    signature = extract_signature_line(code_bytes, child.start_byte)
                    symbols.append(
                        Symbol(
                            file_path="",
                            symbol_name=prop_name,
                            symbol_type="objc_property",
                            language="objc",
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            line_number=child.start_point[0] + 1,
                            signature=signature,
                            inherits=[],
                            conforms=[],
                            extends=None,
                            calls=[],
                        )
                    )
                continue

            if method_node is None:
                continue

            selector = _extract_objc_method_selector(method_node)
            if selector is None:
                continue
            # Bare-keyword (nullary) selector: ``init`` -> name "init",
            # selector "init". Keyword-with-colons selector:
            # ``addValue:toValue:`` -> name "addValue:toValue:" (the
            # whole selector, since ObjC has no name-vs-call-site
            # distinction the way Swift does), selector
            # "addValue:toValue:".
            symbols.append(
                Symbol(
                    file_path="",
                    symbol_name=selector,
                    symbol_type="objc_method",
                    language="objc",
                    start_byte=method_node.start_byte,
                    end_byte=method_node.end_byte,
                    line_number=method_node.start_point[0] + 1,
                    signature=extract_signature_line(code_bytes, method_node.start_byte),
                    inherits=[],
                    conforms=[],
                    extends=None,
                    calls=[],
                    selector=selector,
                )
            )
        # parent_signature_line is unused by the inner loop but kept in
        # the signature to mirror the Swift extractor's structure for
        # future per-row signature overrides.
        _ = parent_signature_line

    def walk(node) -> None:
        node_type = node.type

        if node_type == "preproc_include":
            # ObjC #import <Foundation/Foundation.h> -- record the bare
            # framework name (first path component without extension)
            # so the bridging-edges resolver has something to work
            # with. Defensive: not all includes have a parseable
            # path, so skip when the structure is unusual.
            for child in node.children:
                if child.type in {"system_lib_string", "string_literal"}:
                    raw = child.text.decode("utf-8", errors="ignore").strip("<>\"")
                    head = raw.split("/", 1)[0]
                    head = head.rsplit(".", 1)[0]
                    if head:
                        imports.append(head)
                    break
            return

        if node_type == "class_interface":
            class_name: Optional[str] = None
            category_name: Optional[str] = None
            superclass: Optional[str] = None
            for i, child in enumerate(node.children):
                field = node.field_name_for_child(i)
                if field == "superclass" and child.type == "identifier":
                    superclass = child.text.decode("utf-8", errors="ignore")
                elif field == "category" and child.type == "identifier":
                    category_name = child.text.decode("utf-8", errors="ignore")
                elif (
                    field is None
                    and child.type == "identifier"
                    and class_name is None
                ):
                    class_name = child.text.decode("utf-8", errors="ignore")
            if class_name is None:
                return

            line_number = node.start_point[0] + 1
            signature = extract_signature_line(code_bytes, node.start_byte)

            if category_name is not None:
                key = (class_name, category_name)
                if key not in seen_categories:
                    seen_categories.add(key)
                    symbols.append(
                        Symbol(
                            file_path="",
                            symbol_name=category_name,
                            symbol_type="category",
                            language="objc",
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            line_number=line_number,
                            signature=signature,
                            inherits=[],
                            conforms=[],
                            extends=class_name,
                            calls=[],
                        )
                    )
            else:
                if class_name not in seen_classes:
                    seen_classes.add(class_name)
                    inherits: List[str] = []
                    if superclass:
                        inherits.append(normalize_type_name(superclass))
                    symbols.append(
                        Symbol(
                            file_path="",
                            symbol_name=class_name,
                            symbol_type="objc_class",
                            language="objc",
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            line_number=line_number,
                            signature=signature,
                            inherits=inherits,
                            conforms=[],
                            extends=None,
                            calls=[],
                        )
                    )

            emit_methods(node, line_number)
            return

        if node_type == "class_implementation":
            class_name = None
            category_name = None
            for i, child in enumerate(node.children):
                field = node.field_name_for_child(i)
                if field == "category" and child.type == "identifier":
                    category_name = child.text.decode("utf-8", errors="ignore")
                elif field is None and child.type == "identifier" and class_name is None:
                    class_name = child.text.decode("utf-8", errors="ignore")
            if class_name is None:
                return

            line_number = node.start_point[0] + 1
            signature = extract_signature_line(code_bytes, node.start_byte)

            if category_name is not None:
                key = (class_name, category_name)
                if key not in seen_categories:
                    seen_categories.add(key)
                    symbols.append(
                        Symbol(
                            file_path="",
                            symbol_name=category_name,
                            symbol_type="category",
                            language="objc",
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            line_number=line_number,
                            signature=signature,
                            inherits=[],
                            conforms=[],
                            extends=class_name,
                            calls=[],
                        )
                    )
            else:
                if class_name not in seen_classes:
                    seen_classes.add(class_name)
                    symbols.append(
                        Symbol(
                            file_path="",
                            symbol_name=class_name,
                            symbol_type="objc_class",
                            language="objc",
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            line_number=line_number,
                            signature=signature,
                            inherits=[],
                            conforms=[],
                            extends=None,
                            calls=[],
                        )
                    )

            emit_methods(node, line_number)
            return

        if node_type == "protocol_declaration":
            protocol_name: Optional[str] = None
            for child in node.children:
                if child.type == "identifier":
                    protocol_name = child.text.decode("utf-8", errors="ignore")
                    break
            if protocol_name:
                line_number = node.start_point[0] + 1
                signature = extract_signature_line(code_bytes, node.start_byte)
                conforms: List[str] = []
                for child in node.children:
                    if child.type == "protocol_reference_list":
                        for inner in child.children:
                            if inner.type == "identifier":
                                conforms.append(
                                    normalize_type_name(
                                        inner.text.decode("utf-8", errors="ignore")
                                    )
                                )
                symbols.append(
                    Symbol(
                        file_path="",
                        symbol_name=protocol_name,
                        symbol_type="objc_protocol",
                        language="objc",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        line_number=line_number,
                        signature=signature,
                        inherits=[],
                        conforms=conforms,
                        extends=None,
                        calls=[],
                    )
                )
                emit_methods(node, line_number)
            return

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return symbols, sorted(set(imports))


def fallback_regex_parse(text: str, encoding: str, file_path: str) -> List[Symbol]:
    patterns = [
        ("function", re.compile(r"\bfunc\s+(\w+)\s*\(")),
        ("class", re.compile(r"\bclass\s+(\w+)")),
        ("struct", re.compile(r"\bstruct\s+(\w+)")),
        ("protocol", re.compile(r"\bprotocol\s+(\w+)")),
        ("extension", re.compile(r"\bextension\s+(\w+)")),
    ]
    symbols: List[Symbol] = []
    for symbol_type, pattern in patterns:
        for match in pattern.finditer(text):
            name = match.group(1)
            line_number = text.count("\n", 0, match.start()) + 1
            start_byte = len(text[: match.start()].encode(encoding))
            line_end = text.find("\n", match.start())
            if line_end == -1:
                line_end = len(text)
            end_byte = len(text[:line_end].encode(encoding))
            signature = text[match.start() : line_end].strip()
            symbols.append(
                Symbol(
                    file_path=file_path,
                    symbol_name=name,
                    symbol_type=symbol_type,
                    language="swift" if file_path.endswith(".swift") else "objc",
                    start_byte=start_byte,
                    end_byte=end_byte,
                    line_number=line_number,
                    signature=signature,
                    inherits=[],
                    conforms=[],
                    extends=None,
                    calls=[],
                )
            )
    return symbols


def parse_file(path: str) -> ParseResult:
    try:
        code_bytes, text, encoding = read_file_text(path)
    except Exception as exc:
        log_error(path, f"Read failure: {exc}")
        return ParseResult(file_path=path, symbols=[], imports=[])

    parser = Parser()
    try:
        if path.endswith(".swift"):
            parser.language = Language(tree_sitter_swift.language())
            tree = parser.parse(code_bytes)
            # Note: We proceed even if tree.root_node.has_error because
            # tree-sitter trees with errors often still contain valid declaration nodes
            symbols, imports = extract_swift_symbols(code_bytes, tree)
        else:
            parser.language = Language(tree_sitter_objc.language())
            tree = parser.parse(code_bytes)
            symbols, imports = extract_objc_symbols(code_bytes, tree)
        for symbol in symbols:
            symbol.file_path = path
        return ParseResult(file_path=path, symbols=symbols, imports=imports)
    except Exception as exc:
        log_error(path, f"Tree-sitter parse failure: {exc}")
        fallback_symbols = fallback_regex_parse(text, encoding, path)
        return ParseResult(file_path=path, symbols=fallback_symbols, imports=[])


def ensure_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    """Apply any unapplied schema migrations to ``conn``.

    Phase 6a replaced the previous executescript-of-schema.sql flow with
    a migration runner. ``schema_path`` is no longer read directly --
    ``001_baseline.sql`` is a CREATE-TABLE-IF-NOT-EXISTS rollup of the
    schema, so a fresh DB and a legacy DB (existing tables, no
    schema_version row) both reach the latest version through the same
    code path. The parameter is retained for API compatibility with
    callers/tests that pass it.
    """
    new_version = _migrations.apply_migrations(conn)
    log.info("schema at v%d after migration check", new_version)


def load_hashes(conn):
    try:
        # Try to load existing hashes
        return {row[0]: row[1] for row in conn.execute("SELECT path, hash FROM file_hashes")}
    except sqlite3.OperationalError:
        # Table doesn't exist yet (Fresh DB), so return empty dict implies "Rebuild Everything"
        return {}


def chunked(iterable: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def _load_model() -> "SentenceTransformer":
    """Load (or return the cached) ``SentenceTransformer``.

    Resolves the model path via the documented precedence
    (``$IOS_GRAPHRAG_MODEL_DIR`` → ``~/.cache/ios-graphrag/models`` → HF id),
    plus the legacy in-repo ``models/`` snapshot fallback for backward
    compatibility.

    The loaded model is cached on a module-level singleton so subsequent
    ``embed_signatures`` calls reuse it.  If the resolved model path
    changes between calls (e.g. an env-var override flipped on), the cache
    is invalidated and the model reloaded.

    Phase 4b note: this function runs in the indexer's parent process; any
    TLS posture configured by ``_tls.configure_insecure_tls_if_requested``
    in ``main()`` is already in effect and applies transparently to the
    HF download path.
    """
    global _MODEL_CACHE, _MODEL_CACHE_PATH

    # Resolve the model path/identifier with the documented precedence
    # ($IOS_GRAPHRAG_MODEL_DIR → ~/.cache/ios-graphrag/models → HF id).
    # This is consulted first so an env-var or per-user cache wins over the
    # legacy in-repo `models/` snapshot below.
    model_path = _resolve_model_path()

    # Legacy fallback: if the resolver returned the HF identifier (i.e. no
    # offline override matched) but an in-repo `models/` snapshot exists,
    # prefer that snapshot. Preserves the prior behavior for users who
    # staged the model under the package's own directory.
    if model_path == MODEL_NAME:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_models_dir = os.path.join(script_dir, "..", "models")
        if os.path.isdir(local_models_dir):
            os.environ["HF_HOME"] = os.path.abspath(local_models_dir)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            snapshots_dir = os.path.join(local_models_dir, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    model_path = os.path.join(snapshots_dir, snapshots[0])
                    log.info(f"model loaded from in-repo snapshot: {model_path}")

    if _MODEL_CACHE is not None and _MODEL_CACHE_PATH == model_path:
        return _MODEL_CACHE

    import torch

    model = SentenceTransformer(model_path, trust_remote_code=True)
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
    except Exception:
        pass

    _MODEL_CACHE = model
    _MODEL_CACHE_PATH = model_path
    return model


def embed_signatures(signatures: List[str]) -> List[np.ndarray]:
    """Generate embeddings in-process using a cached SentenceTransformer.

    Phase 4b refactor: previously this spawned a one-shot
    ``ProcessPoolExecutor`` (with a ``multiprocessing.spawn`` context) and
    reloaded the model from disk for every call.  The audit found the
    contamination concerns that motivated that isolation no longer apply
    under the pinned ``sentence-transformers``/``torch``/``huggingface_hub``
    versions, so the model now lives in the parent process and is cached
    on a module-level singleton across calls.
    """
    if not signatures:
        return []

    model = _load_model()

    embeddings: List[np.ndarray] = []
    for i in range(0, len(signatures), BATCH_SIZE):
        batch = signatures[i : i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    return embeddings


def update_file_index(
    conn: sqlite3.Connection,
    file_path: str,
    file_hash: str,
    symbols: List[Symbol],
    use_savepoints: bool,
    savepoint_name: str,
) -> None:
    with transaction_scope(conn, use_savepoints, savepoint_name):
        conn.execute("DELETE FROM nodes WHERE file_path = ?", (file_path,))
        conn.execute(
            "INSERT OR REPLACE INTO file_hashes (path, hash) VALUES (?, ?)",
            (file_path, file_hash),
        )
        if symbols:
            # Phase 4.5d: named-column INSERT to keep this resilient against
            # future column-order skew. Migration 004 (Phase 5 P0's
            # symbol_type CHECK rebuild) explicitly preserved the column
            # order baseline ∪ 002 ∪ migration-004's CHECK extension --
            # using ``INSERT INTO nodes (col, col, ...) VALUES (...)`` here
            # means a future migration that inserts a column at a non-tail
            # position cannot silently mis-bind values.
            conn.executemany(
                """
                INSERT INTO nodes (
                    file_path, symbol_name, symbol_type, language,
                    start_byte, end_byte, line_number, signature, selector,
                    is_swiftui_view, is_observable, state_kind, body_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        symbol.file_path,
                        symbol.symbol_name,
                        symbol.symbol_type,
                        symbol.language,
                        symbol.start_byte,
                        symbol.end_byte,
                        symbol.line_number,
                        symbol.signature,
                        symbol.selector,
                        # SQLite stores Python bool as 0/1 in BOOLEAN columns;
                        # None stays NULL. Cast explicitly so the column reads
                        # back as 1/0/NULL (matching the test predicates).
                        1 if symbol.is_swiftui_view else (None if symbol.is_swiftui_view is None else 0),
                        1 if symbol.is_observable else (None if symbol.is_observable is None else 0),
                        symbol.state_kind,
                        symbol.body_kind,
                    )
                    for symbol in symbols
                ],
            )


def resolve_symbol_ids(conn: sqlite3.Connection, file_paths: List[str]) -> Dict[SymbolKey, int]:
    mapping: Dict[SymbolKey, int] = {}
    for path_chunk in chunked(file_paths, 500):
        placeholders = ",".join("?" for _ in path_chunk)
        query = f"""
        SELECT id, file_path, symbol_name, start_byte, end_byte
        FROM nodes
        WHERE file_path IN ({placeholders})
        """
        for row in conn.execute(query, path_chunk):
            node_id, file_path, symbol_name, start_byte, end_byte = row
            mapping[SymbolKey(file_path, symbol_name, start_byte, end_byte)] = node_id
    return mapping


def resolve_all_symbol_ids(conn: sqlite3.Connection) -> Dict[str, List[int]]:
    """Build a name -> list-of-node-ids map for unresolved-edge fallbacks.

    Determinism (P1.3a): the SELECT is sorted by ``(file_path, start_byte, id)``
    so candidate lists for tie-broken names are reproducible across runs
    regardless of dict insertion order from upstream parsing. Edge resolution
    in ``build_edges_for_file`` and ``update_unresolved_edges`` always picks
    ``target_ids[0]``; with this ordering, "first" is now stable.

    Observability (P1.3b): when the same name maps to >1 node, emits a
    ``log.warning`` with each candidate's ``(file_path, line_number)``. The
    warning travels through the root logger configured in ``main()`` and
    lands in ``indexing_errors.log``.
    """
    name_map: Dict[str, List[int]] = {}
    candidates: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    rows = conn.execute(
        "SELECT id, symbol_name, file_path, line_number "
        "FROM nodes "
        "ORDER BY symbol_name, file_path, start_byte, id"
    )
    for node_id, symbol_name, file_path, line_number in rows:
        name_map.setdefault(symbol_name, []).append(node_id)
        candidates.setdefault(symbol_name, []).append((file_path, line_number))

    for name, ids in name_map.items():
        if len(ids) > 1:
            log.warning(
                "name collision: %r resolves to %d candidates: %s",
                name,
                len(ids),
                candidates[name],
            )

    return name_map


def resolve_selector_ids(conn: sqlite3.Connection) -> Dict[str, int]:
    """Build a map from selector to node ID for function call resolution.

    This enables accurate overload resolution by matching full selectors
    like "test(_:)" instead of just "test".

    Determinism (P1.3a): the SELECT is sorted by ``(file_path, start_byte,
    id)`` so when two functions share a selector, the first one in that
    order wins reproducibly. Without the ORDER BY, a SQLite scan order
    can vary, leaving the picked target dependent on insertion history.

    Observability (P1.3b): when a selector would overwrite an existing
    entry, log a warning naming both the previous and the would-be
    replacement so cross-module collisions are auditable.
    """
    selector_map: Dict[str, int] = {}
    selector_origin: Dict[str, Tuple[str, Optional[int]]] = {}
    collisions: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    rows = conn.execute(
        "SELECT id, selector, file_path, line_number "
        "FROM nodes "
        "WHERE selector IS NOT NULL "
        "ORDER BY selector, file_path, start_byte, id"
    )
    for node_id, selector, file_path, line_number in rows:
        if not selector:
            continue
        if selector in selector_map:
            # Track the collision but keep the original (first by sort key) winner.
            collisions.setdefault(selector, [selector_origin[selector]]).append(
                (file_path, line_number)
            )
            continue
        selector_map[selector] = node_id
        selector_origin[selector] = (file_path, line_number)

    for selector, candidates in collisions.items():
        log.warning(
            "selector collision: %r has %d candidates: %s (kept first; rest ignored)",
            selector,
            len(candidates),
            candidates,
        )

    return selector_map


def build_edges_for_file(
    symbols: List[Symbol],
    imports: List[str],
    id_map: Dict[SymbolKey, int],
    name_map: Dict[str, List[int]],
    selector_map: Optional[Dict[str, int]] = None,
) -> List[Tuple[int, Optional[int], str, str, Optional[int]]]:
    edges: List[Tuple[int, Optional[int], str, str, Optional[int]]] = []

    # Use selector_map for CALLS edges if available, fallback to name_map
    _selector_map = selector_map or {}

    for symbol in symbols:
        source_key = SymbolKey(
            symbol.file_path, symbol.symbol_name, symbol.start_byte, symbol.end_byte
        )
        source_id = id_map.get(source_key)
        if not source_id:
            continue

        for base in symbol.inherits:
            target_ids = name_map.get(base, [])
            edges.append(
                (
                    source_id,
                    target_ids[0] if target_ids else None,
                    base,
                    "INHERITS",
                    symbol.line_number,
                )
            )

        for proto in symbol.conforms:
            target_ids = name_map.get(proto, [])
            edges.append(
                (
                    source_id,
                    target_ids[0] if target_ids else None,
                    proto,
                    "CONFORMS",
                    symbol.line_number,
                )
            )

        if symbol.extends:
            target_ids = name_map.get(symbol.extends, [])
            edges.append(
                (
                    source_id,
                    target_ids[0] if target_ids else None,
                    symbol.extends,
                    "EXTENDS",
                    symbol.line_number,
                )
            )

        for callee in symbol.calls:
            # Use selector_map for precise overload resolution
            # callee is now a full selector like "test(_:)" or "configure(width:height:)"
            target_id = _selector_map.get(callee)
            if target_id is None:
                # Fallback to name-based lookup (extract function name from selector)
                callee_name = callee.split("(")[0] if "(" in callee else callee
                target_ids = name_map.get(callee_name, [])
                target_id = target_ids[0] if target_ids else None
            edges.append((source_id, target_id, callee, "CALLS", symbol.line_number))

        if imports and symbol.symbol_type in {
            "class",
            "struct",
            "protocol",
            "enum",
            "extension",
            "objc_class",
            "objc_protocol",
            "category",
        }:
            for module in imports:
                edges.append((source_id, None, module, "IMPORTS", symbol.line_number))

    return edges


def upsert_edges(
    conn: sqlite3.Connection,
    file_path: str,
    edges: List[Tuple[int, Optional[int], str, str, Optional[int]]],
    use_savepoints: bool,
    savepoint_name: str,
) -> None:
    with transaction_scope(conn, use_savepoints, savepoint_name):
        conn.execute(
            """
            DELETE FROM edges
            WHERE source_node_id IN (SELECT id FROM nodes WHERE file_path = ?)
            """,
            (file_path,),
        )
        if edges:
            # INSERT OR IGNORE so that duplicates produced by build_edges_for_file
            # (e.g. a function that calls the same callee twice on the same line)
            # are silently dropped under the Phase 6c unique-index gate, rather
            # than aborting the per-file transaction with IntegrityError.
            conn.executemany(
                """
                INSERT OR IGNORE INTO edges (source_node_id, target_node_id, target_symbol, edge_type, line_number)
                VALUES (?, ?, ?, ?, ?)
                """,
                edges,
            )


def rebuild_extension_map(
    conn: sqlite3.Connection, use_savepoints: bool, savepoint_name: str
) -> None:
    with transaction_scope(conn, use_savepoints, savepoint_name):
        conn.execute("DELETE FROM extension_map")
        symbols = conn.execute(
            """
            SELECT id, symbol_name
            FROM nodes
            WHERE symbol_type = 'extension'
            """
        ).fetchall()
        name_to_canonical = {
            row[1]: row[0]
            for row in conn.execute(
                """
                SELECT id, symbol_name
                FROM nodes
                WHERE symbol_type IN ('class', 'struct', 'protocol', 'enum')
                """
            )
        }
        data = []
        for ext_id, ext_name in symbols:
            canonical_id = name_to_canonical.get(ext_name)
            data.append((ext_id, canonical_id))
        if data:
            conn.executemany(
                """
                INSERT INTO extension_map (extension_node_id, canonical_node_id)
                VALUES (?, ?)
                """,
                data,
            )


def rebuild_bridging_edges(
    conn: sqlite3.Connection, use_savepoints: bool, savepoint_name: str
) -> None:
    with transaction_scope(conn, use_savepoints, savepoint_name):
        conn.execute("DELETE FROM edges WHERE edge_type = 'BRIDGING'")
        swift_classes = conn.execute(
            """
            SELECT id, symbol_name
            FROM nodes
            WHERE symbol_type = 'class' AND language = 'swift'
            """
        ).fetchall()
        objc_classes = {
            row[1]: row[0]
            for row in conn.execute(
                """
                SELECT id, symbol_name
                FROM nodes
                WHERE symbol_type = 'objc_class'
                """
            )
        }
        inherits_edges = conn.execute(
            """
            SELECT source_node_id, target_symbol, line_number
            FROM edges
            WHERE edge_type = 'INHERITS'
            """
        ).fetchall()
        swift_ids = {row[0] for row in swift_classes}
        data = []
        for source_id, target_symbol, line_number in inherits_edges:
            if source_id not in swift_ids:
                continue
            objc_id = objc_classes.get(target_symbol)
            if objc_id:
                data.append((source_id, objc_id, target_symbol, "BRIDGING", line_number))
        if data:
            # INSERT OR IGNORE for the same reason as upsert_edges: tolerate
            # duplicate-row tuples under the Phase 6c unique-index gate.
            conn.executemany(
                """
                INSERT OR IGNORE INTO edges (source_node_id, target_node_id, target_symbol, edge_type, line_number)
                VALUES (?, ?, ?, ?, ?)
                """,
                data,
            )


def update_unresolved_edges(
    conn: sqlite3.Connection,
    name_map: Dict[str, List[int]],
    use_savepoints: bool,
    savepoint_name: str,
) -> None:
    """Resolve previously-unresolved edges using the deterministic name_map.

    Determinism (P1.3a): the SELECT is sorted by edge id so updates are
    applied in a reproducible order. ``target_ids[0]`` already wins because
    ``resolve_all_symbol_ids`` sorts candidates; this query order makes the
    sequence of UPDATEs themselves stable as well.
    """
    rows = conn.execute(
        "SELECT id, target_symbol FROM edges WHERE target_node_id IS NULL ORDER BY id"
    ).fetchall()
    with transaction_scope(conn, use_savepoints, savepoint_name):
        for edge_id, target_symbol in rows:
            target_ids = name_map.get(target_symbol)
            if target_ids:
                conn.execute(
                    "UPDATE edges SET target_node_id = ? WHERE id = ?",
                    (target_ids[0], edge_id),
                )


def store_embeddings(
    conn: sqlite3.Connection,
    symbols: List[Symbol],
    id_map: Dict[SymbolKey, int],
    use_savepoints: bool,
    savepoint_name: str,
) -> None:
    signatures = [symbol.signature for symbol in symbols]
    embeddings = embed_signatures(signatures)
    with transaction_scope(conn, use_savepoints, savepoint_name):
        for symbol, embedding in zip(symbols, embeddings):
            node_id = id_map.get(
                SymbolKey(symbol.file_path, symbol.symbol_name, symbol.start_byte, symbol.end_byte)
            )
            if node_id is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO node_embeddings (node_id, embedding) VALUES (?, ?)",
                (node_id, embedding.astype(np.float32).tobytes()),
            )


def apply_rename(
    conn: sqlite3.Connection,
    old_path: str,
    new_path: str,
    file_hash: str,
    use_savepoints: bool,
    savepoint_name: str,
) -> None:
    with transaction_scope(conn, use_savepoints, savepoint_name):
        conn.execute(
            "INSERT OR REPLACE INTO file_hashes (path, hash) VALUES (?, ?)",
            (new_path, file_hash),
        )
        conn.execute("UPDATE nodes SET file_path = ? WHERE file_path = ?", (new_path, old_path))
        conn.execute("DELETE FROM file_hashes WHERE path = ?", (old_path,))


def index_repository(repo_root: str, db_path: str, full_reindex: bool = False) -> None:
    start_time = time.time()
    # Repo layout: <repo>/src/ios_graphrag/indexer.py and <repo>/engine/database/schema.sql
    schema_path = Path(__file__).resolve().parents[2] / "engine" / "database" / "schema.sql"

    conn = sqlite3.connect(db_path)
    ensure_schema(conn, schema_path)
    conn.execute("PRAGMA foreign_keys = ON")

    if full_reindex:
        conn.execute("BEGIN IMMEDIATE")
        conn.executescript(
            """
            DELETE FROM node_embeddings;
            DELETE FROM edges;
            DELETE FROM extension_map;
            DELETE FROM nodes;
            DELETE FROM file_hashes;
            """
        )
        conn.commit()  # Commit deletions so load_hashes() sees empty tables

    stored_hashes = load_hashes(conn)
    files = collect_indexable_files(repo_root)
    use_savepoints = full_reindex
    savepoint_counter = 0

    def next_savepoint(prefix: str) -> str:
        nonlocal savepoint_counter
        savepoint_counter += 1
        return f"{prefix}_{savepoint_counter}"

    # Phase boundary markers (DEBUG-level so they're invisible in INFO output
    # but the benchmark harness can scrape them by setting GRAPHRAG_LOG_LEVEL=DEBUG
    # and parsing log lines that match "PHASE_START <name>" / "PHASE_END <name>".
    # Markers are intentionally simple substrings, not structured records.
    log.debug("PHASE_START hash")
    hash_results: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = {executor.submit(compute_hash, f): f for f in files}
        for future in as_completed(futures):
            path = futures[future]
            try:
                file_hash = future.result()
            except Exception as exc:
                log_error(path, f"Hash worker failure: {exc}")
                continue
            if file_hash:
                hash_results[path] = file_hash
    log.debug("PHASE_END hash")

    disk_paths = set(hash_results.keys())
    stored_paths = set(stored_hashes.keys())
    orphan_paths = stored_paths - disk_paths
    new_paths = {path for path in disk_paths if path not in stored_paths}
    changed_paths = {
        path
        for path in disk_paths
        if stored_hashes.get(path) and stored_hashes[path] != hash_results[path]
    }

    hash_to_new_paths: Dict[str, List[str]] = {}
    for path in new_paths:
        hash_to_new_paths.setdefault(hash_results[path], []).append(path)

    for old_path in list(orphan_paths):
        old_hash = stored_hashes.get(old_path)
        candidates = hash_to_new_paths.get(old_hash, [])
        if candidates:
            new_path = candidates.pop(0)
            apply_rename(
                conn, old_path, new_path, old_hash, use_savepoints, next_savepoint("rename")
            )
            new_paths.discard(new_path)
            orphan_paths.discard(old_path)

    for removed_path in orphan_paths:
        with transaction_scope(conn, use_savepoints, next_savepoint("prune")):
            conn.execute("DELETE FROM file_hashes WHERE path = ?", (removed_path,))

    files_to_parse = sorted(new_paths | changed_paths)
    parse_results: List[ParseResult] = []
    if files_to_parse:
        log.debug("PHASE_START parse")
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            futures = {executor.submit(parse_file, path): path for path in files_to_parse}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log_error(path, f"Parse worker failure: {exc}")
                    continue
                parse_results.append(result)
        log.debug("PHASE_END parse")

    file_to_symbols: Dict[str, List[Symbol]] = {res.file_path: res.symbols for res in parse_results}
    file_to_imports: Dict[str, List[str]] = {res.file_path: res.imports for res in parse_results}

    # DEBUG: Trace edge building
    total_symbols_with_inheritance = sum(
        1
        for symbols in file_to_symbols.values()
        for s in symbols
        if s.inherits or s.conforms or s.extends
    )
    log.debug(
        "files_to_parse=%d, parse_results=%d, symbols_with_inheritance=%d",
        len(files_to_parse),
        len(parse_results),
        total_symbols_with_inheritance,
    )

    # P1.4: per-file crash isolation — track failed files so subsequent
    # passes skip them, and so the run summary reports indexed/failed
    # counts. Setup steps (DB connection, schema, hash collection) above
    # are intentionally NOT wrapped — those failures should still abort
    # the run loudly.
    failed_paths: set = set()
    if files_to_parse:
        log.debug("PHASE_START sql_write")
    for file_path in files_to_parse:
        try:
            update_file_index(
                conn,
                file_path,
                hash_results[file_path],
                file_to_symbols.get(file_path, []),
                use_savepoints,
                next_savepoint("update_file"),
            )
        except Exception as exc:
            log.error(
                "Failed to index %s during update_file_index: %s: %s",
                file_path,
                type(exc).__name__,
                exc,
            )
            failed_paths.add(file_path)
    if files_to_parse:
        log.debug("PHASE_END sql_write")

    if files_to_parse:
        log.debug("PHASE_START weave")
        id_map = resolve_symbol_ids(conn, files_to_parse)
        name_map = resolve_all_symbol_ids(conn)
        selector_map = resolve_selector_ids(conn)  # For selector-based CALLS resolution
        log.debug(
            "id_map size=%d, name_map size=%d, selector_map size=%d",
            len(id_map),
            len(name_map),
            len(selector_map),
        )

        total_edges = 0
        for file_path in files_to_parse:
            if file_path in failed_paths:
                continue
            try:
                symbols = file_to_symbols.get(file_path, [])
                edges = build_edges_for_file(
                    symbols,
                    file_to_imports.get(file_path, []),
                    id_map,
                    name_map,
                    selector_map,
                )
                total_edges += len(edges)
                upsert_edges(conn, file_path, edges, use_savepoints, next_savepoint("edges"))
            except Exception as exc:
                log.error(
                    "Failed to build/upsert edges for %s: %s: %s",
                    file_path,
                    type(exc).__name__,
                    exc,
                )
                failed_paths.add(file_path)
        log.debug("Total edges built = %d", total_edges)
        # The late passes operate on the global edges/nodes view; they're
        # not per-file and have no isolation requirement. If they crash,
        # the run aborts (that's fine — they'd indicate a schema or
        # connection-level fault, not a per-file content issue).
        rebuild_extension_map(conn, use_savepoints, next_savepoint("extensions"))
        update_unresolved_edges(conn, name_map, use_savepoints, next_savepoint("resolve_edges"))
        rebuild_bridging_edges(conn, use_savepoints, next_savepoint("bridging"))
        log.debug("PHASE_END weave")

        # Embedding step is batched across all files; a malformed payload
        # from any one file would crash the whole batch. Documented as a
        # known limitation in docs/LIMITATIONS.md (L2). Phase 4 will
        # introduce per-file embedding isolation.
        log.debug("PHASE_START embed")
        all_symbols_to_embed = [
            symbol
            for fp, symbols in file_to_symbols.items()
            if fp not in failed_paths
            for symbol in symbols
        ]
        store_embeddings(
            conn, all_symbols_to_embed, id_map, use_savepoints, next_savepoint("embeddings")
        )
        log.debug("PHASE_END embed")

    if full_reindex:
        conn.commit()

    # Phase 6c integrity check. Fail-soft: any exception in here is logged
    # as a warning by the helper itself and never aborts the indexing run.
    try:
        from . import _integrity  # local import to avoid widening import-time cost

        _integrity.log_integrity_check(conn)
    except Exception as exc:  # noqa: BLE001
        log.warning("integrity check wrapper failed: %s: %s", type(exc).__name__, exc)

    conn.close()
    elapsed = time.time() - start_time
    n_failed = len(failed_paths)
    # n_indexed counts the number of files attempted that completed
    # without raising in either update_file_index or build/upsert edges.
    # Files that fail at parse-worker time are not in `files_to_parse`
    # via parse_results, so they're already excluded from this count.
    n_indexed = len(files_to_parse) - n_failed
    log.info(
        "Indexing complete in %.2fs: %d files indexed, %d files failed",
        elapsed,
        n_indexed,
        n_failed,
    )
    if failed_paths:
        log.warning("Failed files: %s", sorted(failed_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG production indexer")
    parser.add_argument("--repo", required=True, help="Repository root path")
    parser.add_argument("--db", default=DB_DEFAULT, help="SQLite database path")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    parser.add_argument(
        "--cert-bundle",
        metavar="PATH",
        default=None,
        help=(
            "Path to a corporate CA bundle (PEM). Sets REQUESTS_CA_BUNDLE and "
            "SSL_CERT_FILE for the duration of this process; TLS verification "
            "stays ON. Recommended over GRAPHRAG_INSECURE_TLS=1."
        ),
    )
    args = parser.parse_args()

    # Phase 6b: centralized logger setup. Stderr gets human-readable text
    # (kept for the test_indexer_stdout invariant + interactive use); the
    # rotating file handler at ~/Library/Logs/ios-graphrag/indexer.log gets
    # JSON lines for grep/jq/structured analysis. GRAPHRAG_LOG_LEVEL still
    # tunes verbosity (default INFO; DEBUG enables PHASE_* benchmark
    # markers). GRAPHRAG_LOG_DIR overrides the path (used by tests).
    from ._logging import setup_logging

    setup_logging("indexer")

    # TLS configuration must come AFTER logging is wired so the WARNING emitted
    # by configure_insecure_tls_if_requested() actually reaches stderr/log file.
    _tls.configure_insecure_tls_if_requested()
    if args.cert_bundle:
        _tls.configure_cert_bundle(args.cert_bundle)

    index_repository(args.repo, args.db, args.full)


if __name__ == "__main__":
    main()
