# FilmLens - Makefile
# Comandos para desarrollo local

.PHONY: help
help: ## Mostrar ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Setup inicial
.PHONY: setup
setup: ## Crear archivos y directorios necesarios
	@if [ ! -f .env ]; then cp .env.example .env && echo "Archivo .env creado"; fi
	@mkdir -p data/raw data/processed
	@mkdir -p models/production
	@mkdir -p mlruns
	@mkdir -p src/filmlens/trained_models
	@echo "Setup completo"

# Docker services
.PHONY: build
build: ## Build imagenes Docker
	docker-compose build

.PHONY: dev
dev: ## Iniciar servicios (MLflow, PostgreSQL, API)
	docker-compose up -d
	@echo "Servicios iniciados:"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  API:        http://localhost:8000"

.PHONY: down
down: ## Detener servicios
	docker-compose down

.PHONY: restart
restart: down dev ## Reiniciar servicios

.PHONY: logs
logs: ## Ver logs de servicios
	docker-compose logs -f

.PHONY: logs-mlflow
logs-mlflow: ## Ver logs de MLflow
	docker-compose logs -f mlflow

.PHONY: logs-api
logs-api: ## Ver logs de API
	docker-compose logs -f api

# Shells
.PHONY: api-shell
api-shell: ## Shell en contenedor API
	docker-compose exec api bash

# Testing
.PHONY: test
test: ## Ejecutar tests
	pytest tests/ -v

.PHONY: test-coverage
test-coverage: ## Tests con coverage
	pytest tests/ -v --cov=src/filmlens --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# Database migrations
.PHONY: db-migrate
db-migrate: ## Aplicar migraciones Flyway
	docker-compose up -d database
	@sleep 5
	docker-compose run --rm flyway migrate

.PHONY: db-info
db-info: ## Info de migraciones
	docker-compose run --rm flyway info

.PHONY: db-validate
db-validate: ## Validar migraciones
	docker-compose run --rm flyway validate

.PHONY: db-repair
db-repair: ## Reparar tabla de migraciones
	docker-compose run --rm flyway repair

.PHONY: db-connect
db-connect: ## Conectar a PostgreSQL
	docker-compose exec database psql -U filmlens -d triggers

# Limpieza
.PHONY: clean
clean: ## Limpiar archivos temporales
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage

.PHONY: clean-docker
clean-docker: ## Limpiar Docker
	docker-compose down -v
	docker system prune -f

# Info
.PHONY: status
status: ## Status de servicios
	@docker-compose ps

.DEFAULT_GOAL := help
