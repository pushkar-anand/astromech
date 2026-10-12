APP     ?= REPO_NAME
GO      ?= go
TAGS    ?=
GOFLAGS  = $(if $(TAGS),-tags $(TAGS))

.PHONY: help tidy fmt vet lint test cover gen build run clean new_migration

help: ## Show this help
	@grep -hE '^[a-z_]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

tidy: ## Sync go.mod/go.sum
	$(GO) mod tidy

fmt: ## Format the tree (used by the autoformat workflow)
	$(GO) fmt ./... && $(GO) tool goimports -w .

vet: ## Run go vet
	$(GO) vet $(GOFLAGS) ./...

lint: ## Run staticcheck
	$(GO) tool staticcheck $(GOFLAGS) ./...

test: ## Run the test suite with the race detector
	$(GO) test $(GOFLAGS) -race -count=1 ./...

cover: ## Run tests and report coverage
	$(GO) test $(GOFLAGS) -race -count=1 -coverprofile=coverage.out -covermode=atomic ./...
	$(GO) tool cover -func=coverage.out | tail -1

gen: ## Regenerate sqlc models (checked by CI)
	$(GO) generate ./...

build: gen ## Build the binary into bin/
	$(GO) build $(GOFLAGS) -o bin/ ./cmd/$(APP)

run: build ## Build and run
	./bin/$(APP)

clean: ## Remove build and coverage artifacts
	rm -rf bin/ coverage.out

new_migration: ## Create a new migration. Usage: make new_migration name=<migration_name>
	@test -n "$(name)" || { echo "usage: make new_migration name=<migration_name>"; exit 1; }
	$(GO) tool migrate create -dir=internal/db/migrations/ -seq -ext sql $(name)
