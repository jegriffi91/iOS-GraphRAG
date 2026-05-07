-- 006_add_subscript_typealias_symbol_types.sql -- Phase 5 P2: extend
-- nodes.symbol_type to cover Swift subscripts and typealiases AND
-- proper ObjC kinds.
--
-- Why a table rebuild: SQLite cannot ALTER an existing CHECK constraint
-- (or add one to an existing table). The canonical workaround is the
-- same one migrations 004 and 005 used: create a new table with the
-- desired constraint, copy rows over, drop the old table, and rename
-- the new one in its place.
--
-- Pre-existing schema_version after migration 005: v5. The columns
-- being copied are the same 14 columns that 005 froze: union of
-- 001_baseline.sql (10 base columns) and 002_add_swiftui_columns.sql
-- (4 SwiftUI columns), in the order id, file_path, symbol_name,
-- symbol_type, language, start_byte, end_byte, line_number, signature,
-- selector, is_swiftui_view, is_observable, state_kind, body_kind.
-- Columns are listed explicitly in both the CREATE TABLE and the
-- INSERT to make the copy resilient to column-order skew if a future
-- migration inserts a column at a non-trailing position.
--
-- Foreign keys: SQLite disables foreign_keys by default. The indexer
-- enables it via PRAGMA only AFTER apply_migrations() returns, so this
-- rebuild runs with FK enforcement OFF and the references from edges /
-- extension_map / node_embeddings into nodes(id) won't trip during the
-- DROP. (PRAGMA foreign_keys can't be toggled inside a transaction
-- anyway -- it's silently ignored mid-transaction. Don't add such a
-- PRAGMA here; rely on the surrounding indexer behaviour.)
--
-- After this migration:
--   * symbol_type CHECK constraint allows the migration-005 set
--     ('class','struct','protocol','enum','function','extension',
--     'property','computed_property','initializer','deinitializer',
--     'enum_case') plus the new Phase 5 P2 Swift kinds 'subscript'
--     and 'typealias', plus the explicit ObjC kinds 'objc_class',
--     'objc_method', 'objc_property', 'objc_protocol', and 'category'
--     so ObjC declarations no longer overload the generic Swift kind
--     names.
--   * sqlite_sequence for 'nodes' carries over via the RENAME, so
--     future AUTOINCREMENT IDs continue from MAX(id) without reuse.
--   * All existing indexes are recreated against the renamed table.

CREATE TABLE nodes_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL CHECK (symbol_type IN (
        'class',
        'struct',
        'protocol',
        'enum',
        'function',
        'extension',
        'property',
        'computed_property',
        'initializer',
        'deinitializer',
        'enum_case',
        'subscript',
        'typealias',
        'objc_class',
        'objc_method',
        'objc_property',
        'objc_protocol',
        'category'
    )),
    language TEXT NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    line_number INTEGER,
    signature TEXT NOT NULL,
    selector TEXT,
    is_swiftui_view BOOLEAN,
    is_observable BOOLEAN,
    state_kind TEXT,
    body_kind TEXT
);

INSERT INTO nodes_new (
    id, file_path, symbol_name, symbol_type, language,
    start_byte, end_byte, line_number, signature, selector,
    is_swiftui_view, is_observable, state_kind, body_kind
) SELECT
    id, file_path, symbol_name, symbol_type, language,
    start_byte, end_byte, line_number, signature, selector,
    is_swiftui_view, is_observable, state_kind, body_kind
FROM nodes;

DROP TABLE nodes;

ALTER TABLE nodes_new RENAME TO nodes;

CREATE INDEX IF NOT EXISTS idx_nodes_file_path ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_symbol_name ON nodes(symbol_name);
CREATE INDEX IF NOT EXISTS idx_nodes_symbol_type ON nodes(symbol_type);
CREATE INDEX IF NOT EXISTS idx_nodes_selector ON nodes(selector);

INSERT OR REPLACE INTO schema_version (version) VALUES (6);
