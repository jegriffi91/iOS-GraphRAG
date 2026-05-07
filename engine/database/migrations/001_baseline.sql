-- 001_baseline.sql -- initial schema for iOS-GraphRAG.
--
-- For a fresh DB (no existing tables), this creates everything.
-- For a legacy DB (existing tables, no schema_version row), the migration
-- runner detects existing tables and stamps schema_version directly to 1
-- without re-running the DDL -- the CREATE TABLE IF NOT EXISTS guards make
-- the DDL itself a no-op against legacy DBs that already have the tables.

CREATE TABLE IF NOT EXISTS file_hashes (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    language TEXT NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    line_number INTEGER,
    signature TEXT NOT NULL,
    selector TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER,
    target_symbol TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    line_number INTEGER,
    FOREIGN KEY (source_node_id) REFERENCES nodes(id),
    FOREIGN KEY (target_node_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS extension_map (
    extension_node_id INTEGER PRIMARY KEY,
    canonical_node_id INTEGER,
    FOREIGN KEY (extension_node_id) REFERENCES nodes(id),
    FOREIGN KEY (canonical_node_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_file_path ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_symbol_name ON nodes(symbol_name);
CREATE INDEX IF NOT EXISTS idx_nodes_symbol_type ON nodes(symbol_type);
CREATE INDEX IF NOT EXISTS idx_nodes_selector ON nodes(selector);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR REPLACE INTO schema_version (version) VALUES (1);
