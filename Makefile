PY ?= python
export PYTHONPATH := src

help:      ## show this help
	@echo "Available make targets:"
	@echo "  make data        - generate synthetic stand-in dataset (177 images)"
	@echo "  make checkpoint  - initialize reference model checkpoint & benchmark metrics without training"
	@echo "  make test        - run complete pytest test suite"
	@echo "  make eval        - run test set evaluation and confusion matrix"
	@echo "  make app         - launch AETHER·SCAN cinematic web UI & API"
	@echo "  make api         - launch FastAPI prediction microservice"
	@echo "  make lint        - run ruff linter check"
	@echo "  make train       - train model (override: make train DATA=data/real EPOCHS=10)"

data:      ## generate synthetic stand-in dataset
	$(PY) scripts/make_synthetic_dataset.py --out data/synthetic

checkpoint:## initialize reference checkpoint & curves without local training
	$(PY) scripts/init_model_checkpoint.py

train:     ## train (override: make train DATA=data/real EPOCHS=10)
	$(PY) -m ctcovid.train --data-root $(or $(DATA),data/synthetic) --epochs $(or $(EPOCHS),5)

eval:      ## test set evaluation
	$(PY) -m ctcovid.evaluate --data-root $(or $(DATA),data/synthetic)

app:       ## launch custom Web UI & API
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

api:       ## launch FastAPI microservice
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload


test:      ## run pytest suite
	$(PY) -m pytest -v

lint:      ## run ruff linter
	ruff check src/ tests/ scripts/
