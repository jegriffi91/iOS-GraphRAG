# Database migrations

Each migration is a SQL file named `<NNN>_<short_description>.sql` where `NNN`
is a zero-padded numeric version. The migration runner
(`src/ios_graphrag/_migrations.py`) discovers files matching
`^\d{3}_[\w_]+\.sql$`, applies any whose version is greater than the current
`schema_version` value, and runs each migration body inside a transaction so
the version stamp lands atomically with the migration.

## Conventions

- Each migration ends with
  `INSERT OR REPLACE INTO schema_version (version) VALUES (N);` where `N`
  matches the filename prefix.
- DDL inside `001_baseline.sql` uses `CREATE TABLE IF NOT EXISTS` so that
  legacy DBs (which already have the tables but no `schema_version` row) can
  receive the baseline as a no-op stamp instead of erroring on duplicate
  table creation.
- Subsequent migrations can use plain `ALTER TABLE` / `CREATE TABLE`. The
  runner catches `sqlite3.OperationalError` containing "duplicate column"
  and treats the migration as already-applied (handles partial-failure
  recovery).
- Once a migration is committed and shipped, it must never be edited.
  Add a new migration to fix a mistake.

## Runner behaviour

- Migrations apply in numeric order, only those whose version is strictly
  greater than the DB's current `MAX(version)`.
- If the DB version exceeds the highest known migration, the runner raises
  `SchemaMismatchError` -- the deployment is running newer-DB-on-older-code
  and must be upgraded or the index rebuilt.
- The indexer applies migrations on every run (cheap when up to date).
- The server only verifies the version at startup -- it never auto-applies,
  to avoid races with a concurrent indexer process.

## Files

- `001_baseline.sql` -- creates all tables required by the indexer.
  Idempotent on legacy DBs.
- `002_add_swiftui_columns.sql` -- adds nullable columns
  (`is_swiftui_view`, `is_observable`, `state_kind`, `body_kind`) for
  Phase 4.5d SwiftUI enrichment. Columns stay NULL until 4.5d ships.
