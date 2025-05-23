# OpenWearables Makefile
.PHONY: help install install-dev clean test lint format security docs build deploy

# Default target
help:
	@echo "OpenWearables Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  clean           Clean up build artifacts and cache"
	@echo ""
	@echo "Development:"
	@echo "  test            Run test suite"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black and isort"
	@echo "  security        Run security checks"
	@echo "  docs            Build documentation"
	@echo ""
	@echo "Application:"
	@echo "  init            Initialize OpenWearables configuration"
	@echo "  start           Start OpenWearables system"
	@echo "  stop            Stop OpenWearables system"
	@echo "  status          Show system status"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-dev      Start development environment"
	@echo "  docker-prod     Start production environment"
	@echo "  docker-gpu      Start GPU-enabled environment"
	@echo "  docker-clean    Clean Docker resources"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-local    Deploy locally with Docker Compose"
	@echo "  deploy-k8s      Deploy to Kubernetes"
	@echo "  backup          Create data backup"
	@echo "  restore         Restore from backup"

# Setup and Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ docs/_build/

# Development
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=openwearables --cov-report=html --cov-report=term

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

test-performance:
	python -m pytest tests/performance/ -v -m "not slow"

lint:
	python -m flake8 openwearables/
	python -m mypy openwearables/
	python -m pylint openwearables/

format:
	python -m black openwearables/ tests/
	python -m isort openwearables/ tests/

security:
	python -m bandit -r openwearables/
	python -m safety check

docs:
	cd docs && make html

# Application Management
init:
	python -m openwearables.cli.openwearables_cli init

start:
	python -m openwearables.cli.openwearables_cli start

stop:
	python -m openwearables.cli.openwearables_cli stop

status:
	python -m openwearables.cli.openwearables_cli status

health:
	python -m openwearables.cli.openwearables_cli health --days 7

export:
	python -m openwearables.cli.openwearables_cli export --days 30 --format json --output health_data.json

# Docker Commands
docker-build:
	docker build -t openwearables:latest .
	docker build -t openwearables:dev --target development .
	docker build -t openwearables:gpu --target gpu .

docker-dev:
	docker-compose --profile dev up -d openwearables-dev

docker-prod:
	docker-compose up -d

docker-gpu:
	docker-compose --profile gpu up -d openwearables-gpu

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

docker-logs:
	docker-compose logs -f openwearables

# Development Environment
dev-setup: install-dev
	mkdir -p data logs config
	python -m openwearables.cli.openwearables_cli init --force
	@echo "Development environment ready!"

dev-start:
	FLASK_ENV=development FLASK_DEBUG=1 python -m openwearables.ui.app

dev-test-watch:
	python -m pytest tests/ -v --watch

# Data Management
backup:
	@echo "Creating backup..."
	mkdir -p backups
	tar -czf backups/openwearables-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ config/
	@echo "Backup created in backups/"

restore:
	@echo "Available backups:"
	@ls -la backups/*.tar.gz 2>/dev/null || echo "No backups found"
	@echo "To restore: tar -xzf backups/backup-file.tar.gz"

# Monitoring and Maintenance
monitor:
	docker-compose up -d prometheus grafana
	@echo "Monitoring available at:"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000"

logs:
	tail -f logs/openwearables.log

db-migrate:
	python -c "from openwearables.core.architecture import OpenWearablesCore; core = OpenWearablesCore(); print('Database migration completed')"

# Deployment
deploy-local: docker-build
	docker-compose up -d
	@echo "OpenWearables deployed locally"
	@echo "Access at: http://localhost"

deploy-staging:
	@echo "Deploying to staging environment..."
	docker-compose -f docker-compose.staging.yml up -d

deploy-production:
	@echo "Deploying to production environment..."
	@echo "Ensure you have proper backups before proceeding!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ]
	docker-compose -f docker-compose.prod.yml up -d

# Kubernetes Deployment
k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/ingress.yaml

k8s-status:
	kubectl get pods -n openwearables
	kubectl get services -n openwearables

k8s-logs:
	kubectl logs -f deployment/openwearables -n openwearables

k8s-clean:
	kubectl delete namespace openwearables

# Performance Testing
benchmark:
	python -m pytest tests/performance/ -v --benchmark-only

profile:
	python -m cProfile -o profile.stats -m openwearables.cli.openwearables_cli status
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Release Management
version-patch:
	bump2version patch

version-minor:
	bump2version minor

version-major:
	bump2version major

release: test lint security
	@echo "Creating release..."
	python setup.py sdist bdist_wheel
	@echo "Release artifacts created in dist/"

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*

# CI/CD Helpers
ci-test: install-dev test lint security

ci-build: clean
	python setup.py sdist bdist_wheel

ci-deploy: ci-test ci-build
	@echo "CI/CD pipeline completed successfully"

# Development Utilities
shell:
	python -c "import openwearables; from openwearables.core.architecture import OpenWearablesCore; core = OpenWearablesCore(); print('OpenWearables shell ready. Use `core` variable.'); import IPython; IPython.embed()"

jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Database Operations
db-shell:
	sqlite3 data/wearables.db

db-backup:
	cp data/wearables.db backups/wearables-$(shell date +%Y%m%d-%H%M%S).db

db-reset:
	rm -f data/wearables.db
	python -m openwearables.cli.openwearables_cli init --force

# Quick Start for New Users
quickstart: install init
	@echo "OpenWearables Quick Start Complete!"
	@echo "====================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Start the system:    make start"
	@echo "  2. Open web interface:  http://localhost:5000"
	@echo "  3. Check status:        make status"
	@echo "  4. View logs:           make logs"
	@echo ""
	@echo "For development:"
	@echo "  make dev-setup"
	@echo "  make dev-start" 