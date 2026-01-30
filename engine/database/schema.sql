-- Production Schema for iOS-GraphRAG (Enterprise Edition)
-- This schema MUST match the columns used by indexer_prod.py

CREATE TABLE IF NOT EXISTS file_hashes (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,          -- "class", "struct", "protocol", "extension", "function", "enum"
    language TEXT NOT NULL,              -- "swift" | "objc"
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    line_number INTEGER,
    signature TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER,             -- Can be NULL if unresolved
    target_symbol TEXT NOT NULL,        -- The name of the target symbol
    edge_type TEXT NOT NULL,            -- "INHERITS", "CONFORMS", "EXTENDS", "CALLS", "IMPORTS", "BRIDGING"
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
    embedding BLOB NOT NULL,            -- Serialized fp32 numpy array
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_nodes_file_path ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_symbol_name ON nodes(symbol_name);
CREATE INDEX IF NOT EXISTS idx_nodes_symbol_type ON nodes(symbol_type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);