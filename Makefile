GO      ?= go
TAGS    ?=
GOFLAGS  = $(if $(TAGS),-tags $(TAGS))

.PHONY: help tidy fmt vet lint test cover clean

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

clean: ## Remove coverage artifacts
	rm -f coverage.out
