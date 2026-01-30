-- The Map: Where things are, not what they are.
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,          -- "file_path::SymbolName"
    name TEXT,
    type TEXT,                    -- "class", "extension", "protocol"
    file_path TEXT,
    start_byte INTEGER,           -- Pointer to Territory
    end_byte INTEGER,             -- Pointer to Territory
    signature TEXT,               -- The "Skeleton" for context
    vector BLOB                   -- Serialized fp32 numpy array
);

CREATE TABLE IF NOT EXISTS edges (
    source TEXT,
    target TEXT,
    relation TEXT,                -- "INHERITS", "EXTENDS", "CALLS"
    PRIMARY KEY (source, target, relation)
);

CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);