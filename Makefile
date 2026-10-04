.PHONY: install data validate test hardware train evaluate evaluate-test live molab offline-bundle

install:
	python -m pip install -e '.[train,live,dev]'

data:
	bash scripts/download_all.sh

validate:
	python -m btc_predictor.data.validate data/processed

test:
	pytest

hardware:
	btc-hardware --output runs/hardware.json

train:
	bash scripts/train_all.sh

evaluate:
	bash scripts/evaluate_all.sh

evaluate-test:
	bash scripts/evaluate_test.sh

live:
	python -m btc_predictor.live.runner --checkpoint checkpoints/best.pt --bootstrap data/processed

molab:
	marimo run notebooks/molab_live.py

offline-bundle:
	bash scripts/package_offline.sh
