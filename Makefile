PYTHON ?= python

.PHONY: install all baseline bert compress distill report full serve test lint format docker

install:  ## Pinned dependencies + this package (editable)
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

all:  ## Whole pipeline, quick CPU settings
	$(PYTHON) -m newsclf all

baseline bert compress distill report:
	$(PYTHON) -m newsclf $@

full:  ## 20k training articles, full 7.6k test set (hours on CPU)
	$(PYTHON) -m newsclf all --full

serve:  ## NEWSCLF_BACKEND=linear|transformer
	uvicorn newsclf.serve:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

docker:
	docker build -t newsclf .
