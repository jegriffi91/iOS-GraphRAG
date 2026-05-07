-- 002_add_swiftui_columns.sql -- adds columns for Phase 4.5d SwiftUI enrichment.
-- These columns are nullable; the current indexer doesn't populate them.
-- Phase 4.5d will add extraction logic that fills them in.
--
-- The runner is defensive about ALTER TABLE failures: if a column already
-- exists (a previously-partial migration), sqlite3 raises OperationalError
-- with "duplicate column name" and the runner treats it as already-applied.

ALTER TABLE nodes ADD COLUMN is_swiftui_view BOOLEAN;
ALTER TABLE nodes ADD COLUMN is_observable BOOLEAN;
ALTER TABLE nodes ADD COLUMN state_kind TEXT;
ALTER TABLE nodes ADD COLUMN body_kind TEXT;

INSERT OR REPLACE INTO schema_version (version) VALUES (2);
