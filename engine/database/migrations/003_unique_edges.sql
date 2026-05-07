-- 003_unique_edges.sql -- edge dedup via UNIQUE INDEX (Phase 6c).
--
-- The roadmap calls for UNIQUE(source_node_id, target_symbol, edge_type,
-- line_number) on the edges table. SQLite's ALTER TABLE doesn't support
-- adding a UNIQUE constraint to an existing table, so this migration:
--   1. dedupes existing rows (keeps the lowest rowid per duplicate group)
--   2. creates a UNIQUE INDEX over the four columns (functionally
--      equivalent to a UNIQUE table constraint)
--
-- Schema reality (verified against schema.sql / 001_baseline.sql):
-- the edges table has BOTH target_node_id (nullable, populated post-
-- resolution) AND target_symbol (TEXT NOT NULL, the literal name parsed
-- from source). The roadmap's unique-key spec uses target_symbol, which
-- aligns with the actual column set, so this migration uses target_symbol
-- as the roadmap dictates. target_node_id is NOT in the unique key
-- because (a) it's filled in later by the resolver, so an edge inserted
-- pre-resolution and again post-resolution would falsely be "different",
-- and (b) target_symbol uniquely identifies the textual referent.
--
-- NULL handling: SQLite treats NULL values as DISTINCT in unique
-- indexes, so two edges that share (source, target_symbol, edge_type)
-- but both have NULL line_number will NOT collide. That's acceptable
-- for the current indexer (line_number on edges is taken from the
-- declaring symbol and is rarely NULL in practice).
--
-- Future inserts that would create a duplicate will fail with
-- "UNIQUE constraint failed: ...". The indexer's INSERT calls have
-- been updated to use INSERT OR IGNORE so per-file rebuilds tolerate
-- the case where build_edges_for_file emits identical tuples (e.g. a
-- function that calls the same callee twice on the same line).

-- Step 1: dedupe existing rows.
DELETE FROM edges
WHERE rowid NOT IN (
    SELECT MIN(rowid)
    FROM edges
    GROUP BY source_node_id, target_symbol, edge_type, line_number
);

-- Step 2: enforce uniqueness going forward.
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
    ON edges(source_node_id, target_symbol, edge_type, line_number);

INSERT OR REPLACE INTO schema_version (version) VALUES (3);
