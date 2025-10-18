.PHONY: help install test clean analyze validate process

# Default target
help:
	@echo "AI Text Detection Dataset Preprocessor - Available Commands:"
	@echo ""
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install package in development mode"
	@echo "  make test            - Run tests"
	@echo "  make analyze         - Analyze raw data"
	@echo "  make validate        - Validate processed data"
	@echo "  make process         - Process train.csv (basic)"
	@echo "  make process-all     - Process all raw data files"
	@echo "  make clean           - Remove processed data and cache"
	@echo "  make clean-all       - Remove all generated files including .venv"
	@echo "  make structure       - Show project structure"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e .

# Testing
test:
	python tests/test_preprocess.py

# Data operations
analyze:
	python scripts/analyze_data.py

validate:
	python scripts/validate_processed_data.py

# Processing
process:
	python src/preprocessor/preprocess.py \
		--input data/raw/train.csv \
		--output data/processed

process-all:
	@echo "Processing all datasets..."
	@for file in data/raw/*.csv; do \
		echo "Processing $$file..."; \
		python src/preprocessor/preprocess.py \
			--input "$$file" \
			--output data/processed; \
	done

# Validation mode
validate-schema:
	python src/preprocessor/preprocess.py \
		--input data/raw/train.csv \
		--output data/processed \
		--validate-only

# Cleanup
clean:
	rm -rf data/processed
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf .venv
	rm -rf *.egg-info
	rm -rf build dist

# Project info
structure:
	@echo "Project Structure:"
	@tree -L 3 -I '__pycache__|*.pyc|.venv|*.egg-info' . || \
		find . -not -path '*/\.*' -not -path '*/__pycache__/*' -not -path '*/.venv/*' | head -50
