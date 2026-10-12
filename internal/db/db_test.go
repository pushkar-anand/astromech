package db

import (
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/REPO_NAME/internal/db/models"
)

// newTestDB opens a database in a directory scoped to the test.
func newTestDB(t *testing.T) *DB {
	t.Helper()

	db, err := New(&Config{Path: t.TempDir(), Name: "test.db"})
	require.NoError(t, err)
	require.NotNil(t, db)

	t.Cleanup(func() { _ = db.Conn.Close() })

	return db
}

func TestNew(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	db, err := New(&Config{Path: dir, Name: "test.db"})
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Conn.Close() })

	require.NoError(t, db.Conn.Ping())
	assert.FileExists(t, filepath.Join(dir, "test.db"))
}

func TestNewFailsOnUnwritablePath(t *testing.T) {
	t.Parallel()

	db, err := New(&Config{Path: filepath.Join(t.TempDir(), "no-such-dir"), Name: "test.db"})

	require.Error(t, err)
	assert.Nil(t, db)
}

// latestMigrationVersion is the sequence number of the highest migration in
// migrations/, derived from the files themselves so that adding one does not
// need this test updated.
func latestMigrationVersion(t *testing.T) int {
	t.Helper()

	entries, err := fs.ReadDir(migrationFiles, migrationDir)
	require.NoError(t, err)

	latest := 0

	for _, e := range entries {
		seq, _, ok := strings.Cut(e.Name(), "_")
		require.True(t, ok, "migration %q is not named <seq>_<name>.<dir>.sql", e.Name())

		n, err := strconv.Atoi(seq)
		require.NoError(t, err, "migration %q has a non-numeric sequence", e.Name())

		latest = max(latest, n)
	}

	require.NotZero(t, latest, "no migrations found")

	return latest
}

// TestNewRunsMigrations checks that New applies every migration, with the
// tables the queries expect.
func TestNewRunsMigrations(t *testing.T) {
	t.Parallel()

	db := newTestDB(t)

	var version int
	var dirty bool

	err := db.Conn.QueryRow(`SELECT version, dirty FROM schema_migrations`).Scan(&version, &dirty)
	require.NoError(t, err)

	assert.Equal(t, latestMigrationVersion(t), version)
	assert.False(t, dirty, "migrations left the schema in a dirty state")

	var name string
	err = db.Conn.QueryRow(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'users'`).Scan(&name)
	require.NoError(t, err, "users table was not created")
	assert.Equal(t, "users", name)
}

// TestNewIsIdempotent covers reopening an already-migrated database, which is
// what every restart after the first one does.
func TestNewIsIdempotent(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	cfg := &Config{Path: dir, Name: "test.db"}

	first, err := New(cfg)
	require.NoError(t, err)
	require.NoError(t, first.Conn.Close())

	second, err := New(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = second.Conn.Close() })

	require.NoError(t, second.Conn.Ping())
}

func TestMigrateDBIsIdempotent(t *testing.T) {
	t.Parallel()

	db := newTestDB(t)

	// New already migrated; running it again must be a no-op rather than an error.
	require.NoError(t, migrateDB(db))
}

// TestCreateUser exercises the generated queries against a real migrated
// schema, which is the only place the two are checked against each other.
func TestCreateUser(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	q := models.New(newTestDB(t).Conn)

	user, err := q.CreateUser(ctx, models.CreateUserParams{
		Username:     "ada",
		PasswordHash: "hash",
	})
	require.NoError(t, err)

	assert.NotZero(t, user.ID)
	assert.Equal(t, "ada", user.Username)
	assert.Equal(t, "hash", user.PasswordHash)
	assert.True(t, user.CreatedAt.Valid, "created_at default was not applied")
	assert.NotEmpty(t, user.CreatedAt.String)
}

func TestCreateUserRejectsDuplicateUsername(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	q := models.New(newTestDB(t).Conn)

	params := models.CreateUserParams{Username: "ada", PasswordHash: "hash"}

	_, err := q.CreateUser(ctx, params)
	require.NoError(t, err)

	_, err = q.CreateUser(ctx, params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "UNIQUE constraint failed")
}
