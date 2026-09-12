.PHONY: install test cov lint types check bench example clean

install:
	pip install -e ".[dev]"

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

lint:
	ruff check .

types:
	mypy

check: lint types cov

bench:
	python examples/benchmark.py

example:
	python examples/email_pipeline.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov dist build *.egg-info .coverage
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name '*.db*' -delete
