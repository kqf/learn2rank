.PHONY: all 

all: model/*.py
	@# Check endpoints and install if needed
	@which pointwise > /dev/null || pip install -e .
	pointwise --path data/dataset_v2.csv
