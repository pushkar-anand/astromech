package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/pushkar-anand/build-with-go/config"
	"github.com/pushkar-anand/build-with-go/logger"

	"github.com/pushkar-anand/REPO_NAME/internal/db"
	"github.com/pushkar-anand/REPO_NAME/internal/server"
)

// Link sqlc with go generate, now we need to just run go generate to generate models and functions for DB
//go:generate go tool sqlc generate -f ./../../sqlc.yaml

func main() {
	// run holds every deferred cleanup; main only turns its error into an exit
	// code, so os.Exit never skips a defer.
	if err := run(); err != nil {
		slog.Error("exiting", logger.Err(err))
		os.Exit(1)
	}
}

func run() error {
	ctx := context.Background()

	// Create a context that will be canceled when the OS sends a signal to the process.
	// This will be used to gracefully shut down the application, shutting down the server and other workers.
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGINT, syscall.SIGABRT, syscall.SIGTERM)
	defer cancel()

	cfg, err := config.Load[Config](
		config.WithDefaults(defaults),
		config.WithYAML("REPO_NAME.yaml"),
		config.WithEnvPrefix("REPO_NAME_UPPER_"),
	)
	if err != nil {
		return err
	}

	log := logger.New(
		logger.WithLevel(cfg.Logger.SlogLevel()),
		logger.WithFormat(cfg.Logger.FormatValue()),
	)

	database, err := db.New(&db.Config{Path: cfg.DB.Path, Name: cfg.DB.Name})
	if err != nil {
		return err
	}

	defer func() {
		if cErr := database.Conn.Close(); cErr != nil {
			log.ErrorContext(ctx, "failed to close database", logger.Err(cErr))
		}
	}()

	return server.Start(ctx, &server.Config{
		Addr:   cfg.Server.Host,
		Port:   cfg.Server.Port,
		Logger: log,
	})
}
