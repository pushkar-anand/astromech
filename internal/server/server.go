package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/pushkar-anand/build-with-go/http/server"
	"github.com/pushkar-anand/build-with-go/logger"

	"github.com/pushkar-anand/REPO_NAME/internal/api"
	"github.com/pushkar-anand/REPO_NAME/internal/web"
)

type Config struct {
	Port   int
	Addr   string
	Logger *slog.Logger
}

func Start(ctx context.Context, cfg *Config) error {
	ap := api.NewHandler(cfg.Logger)

	wh, err := web.NewHandler(cfg.Logger)
	if err != nil {
		return fmt.Errorf("failed to init web handler: %w", err)
	}

	mux := http.NewServeMux()

	mux.Handle("/api/", http.StripPrefix("/api", ap))
	mux.Handle("/", wh)

	h := logger.NewHTTPLogger(cfg.Logger)(mux)

	srv := server.New(
		h,
		server.WithLogger(cfg.Logger),
		server.WithHostPort(cfg.Addr, cfg.Port),
	)

	return srv.Serve(ctx)
}
