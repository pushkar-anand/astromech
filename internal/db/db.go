package db

import (
	"database/sql"
	"fmt"
	"path"

	// Registers the "sqlite3" driver with database/sql. golang-migrate's sqlite3
	// driver imports it too, but relying on that would break the moment the
	// migration setup changes, so the dependency is made explicit here.
	//
	// It is cgo-backed: builds and tests need CGO_ENABLED=1.
	_ "github.com/mattn/go-sqlite3"
)

type (
	DB struct {
		Conn *sql.DB
	}

	Config struct {
		Name string
		Path string
	}
)

func New(cfg *Config) (*DB, error) {
	n := path.Join(cfg.Path, cfg.Name)

	conn, err := sql.Open("sqlite3", n)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	db := &DB{
		Conn: conn,
	}

	err = migrateDB(db)
	if err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("failed to migrate database: %w", err)
	}

	return db, nil
}
