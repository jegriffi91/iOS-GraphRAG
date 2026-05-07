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
import ssl

try: 
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_HTTPX'] = '1'

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser
import tree_sitter_objc
import tree_sitter_swift

DB_DEFAULT = "knowledge-graph.sqlite"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 256
MAX_SYMLINK_DEPTH = 10
LOG_PATH = "indexing_errors.log"

BLOCK_DIRS = {"Pods", "Carthage", "DerivedData", ".build", "vendor"}
ALLOWED_EXTS = {".swift", ".h", ".m"}

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Module-level logger. Levels and additional handlers (stderr) are configured
# in main(); when imported as a library the existing WARNING+ file logging
# above is preserved so log_error() still flushes to indexing_errors.log.
log = logging.getLogger(__name__)


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
        if param_child.type == "_" or (
            param_child.type == "simple_identifier" and text == "_"
        ):
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
                                            arg_labels.append(label_child.text.decode("utf-8", errors="ignore") + ":")
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
                            types.append(normalize_type_name(inner.text.decode("utf-8", errors="ignore")))
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
                    )
                )
            # Still recurse to find nested types
            for child in node.children:
                if child.type == "class_body":
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
                    )
                )
            return
        
        # Recurse into children
        for child in node.children:
            walk_node(child)
    
    # Start walking from root
    walk_node(tree.root_node)
    
    return symbols, sorted(set(imports))




def extract_objc_symbols(code_bytes: bytes, tree) -> Tuple[List[Symbol], List[str]]:
    parser = Parser()
    parser.language = Language(tree_sitter_objc.language())
    symbols: List[Symbol] = []

    query_scm = """
    (interface_declaration name: (identifier) @name) @decl
    (protocol_declaration name: (identifier) @name) @decl
    (implementation_definition name: (identifier) @name) @decl
    (method_definition (identifier) @name) @decl
    (property_declaration (identifier) @name) @decl
    """
    try:
        query = parser.language.query(query_scm)
    except Exception as exc:
        log_error("objc_query", f"Query error: {exc}")
        return symbols, []

    captures = query.captures(tree.root_node)
    for node, tag in captures:
        if tag != "name":
            continue
        parent = node.parent
        symbol_type = {
            "interface_declaration": "class",
            "protocol_declaration": "protocol",
            "implementation_definition": "class",
            "method_definition": "function",
            "property_declaration": "property",
        }.get(parent.type)
        if not symbol_type:
            continue
        symbol_name = node.text.decode("utf-8", errors="ignore")
        signature = extract_signature_line(code_bytes, parent.start_byte)
        line_number = parent.start_point[0] + 1
        inherits: List[str] = []
        conforms: List[str] = []
        extends = None
        if parent.type == "interface_declaration":
            for child in parent.children:
                if child.type == "superclass":
                    inherits.append(
                        normalize_type_name(child.text.decode("utf-8", errors="ignore").strip(": "))
                    )
        symbols.append(
            Symbol(
                file_path="",
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                language="objc",
                start_byte=parent.start_byte,
                end_byte=parent.end_byte,
                line_number=line_number,
                signature=signature,
                inherits=inherits,
                conforms=conforms,
                extends=extends,
                calls=[],
            )
        )

    return symbols, []


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
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='file_hashes'"
    )
    if cursor.fetchone():
        return
    conn.executescript(schema_path.read_text(encoding="utf-8"))


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


def _generate_embeddings_worker(signatures: List[str]) -> List[np.ndarray]:
    """Worker function to generate embeddings in a separate process.
    
    This is isolated to ensure a fresh httpx/huggingface_hub client state,
    avoiding 'client has been closed' errors.
    """
    if not signatures:
        return []
    
    # Set up SSL bypass for corporate proxy environments
    import ssl
    import os
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # HuggingFace/httpx specific SSL bypass
    os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['HTTPX_SSL_VERIFY'] = '0'
    
    # Use local model cache and offline mode to avoid network access
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_models_dir = os.path.join(script_dir, '..', 'models')
    model_path = MODEL_NAME  # Default to HF model ID
    
    # Check for local snapshot and use it directly to avoid any network calls
    if os.path.isdir(local_models_dir):
        os.environ['HF_HOME'] = os.path.abspath(local_models_dir)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Find the actual snapshot directory
        snapshots_dir = os.path.join(local_models_dir, 'snapshots')
        if os.path.isdir(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                # Use the first snapshot found (there's typically only one)
                model_path = os.path.join(snapshots_dir, snapshots[0])
    
    # Re-import to ensure clean state in worker
    from sentence_transformers import SentenceTransformer
    import torch
    
    model = SentenceTransformer(model_path, trust_remote_code=True)
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
    except Exception:
        pass

    embeddings: List[np.ndarray] = []
    for i in range(0, len(signatures), BATCH_SIZE):
        batch = signatures[i : i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    return embeddings


def embed_signatures(signatures: List[str]) -> List[np.ndarray]:
    """Generate embeddings using a separate process to avoid httpx conflicts."""
    if not signatures:
        return []
    
    import multiprocessing
    # Use 'spawn' context to create a truly fresh process (not fork)
    # This avoids inheriting broken httpx client state from parent process
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(_generate_embeddings_worker, signatures)
        return future.result()


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
            conn.executemany(
                """
                INSERT INTO nodes (
                    file_path, symbol_name, symbol_type, language,
                    start_byte, end_byte, line_number, signature, selector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    name_map: Dict[str, List[int]] = {}
    for row in conn.execute("SELECT id, symbol_name FROM nodes"):
        node_id, symbol_name = row
        name_map.setdefault(symbol_name, []).append(node_id)
    return name_map


def resolve_selector_ids(conn: sqlite3.Connection) -> Dict[str, int]:
    """Build a map from selector to node ID for function call resolution.
    
    This enables accurate overload resolution by matching full selectors
    like "test(_:)" instead of just "test".
    """
    selector_map: Dict[str, int] = {}
    for row in conn.execute("SELECT id, selector FROM nodes WHERE selector IS NOT NULL"):
        node_id, selector = row
        if selector:
            selector_map[selector] = node_id
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
        source_key = SymbolKey(symbol.file_path, symbol.symbol_name, symbol.start_byte, symbol.end_byte)
        source_id = id_map.get(source_key)
        if not source_id:
            continue

        for base in symbol.inherits:
            target_ids = name_map.get(base, [])
            edges.append((source_id, target_ids[0] if target_ids else None, base, "INHERITS", symbol.line_number))

        for proto in symbol.conforms:
            target_ids = name_map.get(proto, [])
            edges.append((source_id, target_ids[0] if target_ids else None, proto, "CONFORMS", symbol.line_number))

        if symbol.extends:
            target_ids = name_map.get(symbol.extends, [])
            edges.append(
                (source_id, target_ids[0] if target_ids else None, symbol.extends, "EXTENDS", symbol.line_number)
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

        if imports and symbol.symbol_type in {"class", "struct", "protocol", "enum", "extension"}:
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
            conn.executemany(
                """
                INSERT INTO edges (source_node_id, target_node_id, target_symbol, edge_type, line_number)
                VALUES (?, ?, ?, ?, ?)
                """,
                edges,
            )


def rebuild_extension_map(conn: sqlite3.Connection, use_savepoints: bool, savepoint_name: str) -> None:
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


def rebuild_bridging_edges(conn: sqlite3.Connection, use_savepoints: bool, savepoint_name: str) -> None:
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
                WHERE symbol_type = 'class' AND language = 'objc'
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
            conn.executemany(
                """
                INSERT INTO edges (source_node_id, target_node_id, target_symbol, edge_type, line_number)
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
    rows = conn.execute(
        "SELECT id, target_symbol FROM edges WHERE target_node_id IS NULL"
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
            node_id = id_map.get(SymbolKey(symbol.file_path, symbol.symbol_name, symbol.start_byte, symbol.end_byte))
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

    disk_paths = set(hash_results.keys())
    stored_paths = set(stored_hashes.keys())
    orphan_paths = stored_paths - disk_paths
    new_paths = {path for path in disk_paths if path not in stored_paths}
    changed_paths = {path for path in disk_paths if stored_hashes.get(path) and stored_hashes[path] != hash_results[path]}

    hash_to_new_paths: Dict[str, List[str]] = {}
    for path in new_paths:
        hash_to_new_paths.setdefault(hash_results[path], []).append(path)

    for old_path in list(orphan_paths):
        old_hash = stored_hashes.get(old_path)
        candidates = hash_to_new_paths.get(old_hash, [])
        if candidates:
            new_path = candidates.pop(0)
            apply_rename(conn, old_path, new_path, old_hash, use_savepoints, next_savepoint("rename"))
            new_paths.discard(new_path)
            orphan_paths.discard(old_path)

    for removed_path in orphan_paths:
        with transaction_scope(conn, use_savepoints, next_savepoint("prune")):
            conn.execute("DELETE FROM file_hashes WHERE path = ?", (removed_path,))

    files_to_parse = sorted(new_paths | changed_paths)
    parse_results: List[ParseResult] = []
    if files_to_parse:
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

    file_to_symbols: Dict[str, List[Symbol]] = {res.file_path: res.symbols for res in parse_results}
    file_to_imports: Dict[str, List[str]] = {res.file_path: res.imports for res in parse_results}

    # DEBUG: Trace edge building
    total_symbols_with_inheritance = sum(
        1 for symbols in file_to_symbols.values() 
        for s in symbols 
        if s.inherits or s.conforms or s.extends
    )
    log.debug(
        "files_to_parse=%d, parse_results=%d, symbols_with_inheritance=%d",
        len(files_to_parse),
        len(parse_results),
        total_symbols_with_inheritance,
    )

    for file_path in files_to_parse:
        update_file_index(
            conn,
            file_path,
            hash_results[file_path],
            file_to_symbols.get(file_path, []),
            use_savepoints,
            next_savepoint("update_file"),
        )

    if files_to_parse:
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
            symbols = file_to_symbols.get(file_path, [])
            edges = build_edges_for_file(symbols, file_to_imports.get(file_path, []), id_map, name_map, selector_map)
            total_edges += len(edges)
            upsert_edges(conn, file_path, edges, use_savepoints, next_savepoint("edges"))
        log.debug("Total edges built = %d", total_edges)
        rebuild_extension_map(conn, use_savepoints, next_savepoint("extensions"))
        update_unresolved_edges(conn, name_map, use_savepoints, next_savepoint("resolve_edges"))
        rebuild_bridging_edges(conn, use_savepoints, next_savepoint("bridging"))

        all_symbols_to_embed = [symbol for symbols in file_to_symbols.values() for symbol in symbols]
        store_embeddings(conn, all_symbols_to_embed, id_map, use_savepoints, next_savepoint("embeddings"))

    if full_reindex:
        conn.commit()
    conn.close()
    elapsed = time.time() - start_time
    log.info("Indexing complete in %.2fs", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG production indexer")
    parser.add_argument("--repo", required=True, help="Repository root path")
    parser.add_argument("--db", default=DB_DEFAULT, help="SQLite database path")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    args = parser.parse_args()

    # Reconfigure logging for CLI usage:
    #   - INFO level by default (set GRAPHRAG_LOG_LEVEL=DEBUG for the verbose
    #     trace messages that previously went to stdout via print()).
    #   - Stream to stderr so stdout stays clean (defensive: if the indexer is
    #     ever invoked from the MCP server stdio context, leaking to stdout
    #     would corrupt the protocol stream).
    #   - Append to indexing_errors.log (replacing the module-level WARNING
    #     handler installed at import time so we don't double-log every line).
    import sys as _sys

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    level = getattr(logging, os.environ.get("GRAPHRAG_LOG_LEVEL", "INFO").upper(), logging.INFO)

    root = logging.getLogger()
    # Detach any handlers installed at module import so we don't double-emit
    # (the module-level basicConfig adds one FileHandler with a different
    # format; we replace it with our own to keep output consistent).
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    root.setLevel(level)

    stderr_handler = logging.StreamHandler(_sys.stderr)
    stderr_handler.setFormatter(fmt)
    stderr_handler.setLevel(level)
    root.addHandler(stderr_handler)

    file_handler = logging.FileHandler(LOG_PATH)  # opens append mode by default
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    index_repository(args.repo, args.db, args.full)


if __name__ == "__main__":
    main()


