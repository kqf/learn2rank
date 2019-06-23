.PHONY: all 

all: model/*.py
	@# Check endpoints and install if needed
	@which pointwise > /dev/null || pip install -e .
	pointwise --path data/dataset_v2.csv
	@echo "\n\n\nNow testing the pairwise approach\n\n\n"
	pairwise --path data/dataset_v2.csv
