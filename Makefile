PYTHON = python

# ====== Directories ======
VIS_DIR = visualizations


# ====== Basic Setup ======
install:
	$(PYTHON) -m pip install -r requirements.txt

# ====== Data (UrbanSound8K) ======
data:
	$(PYTHON) data/download.py

# ====== Training ======
train_resnet: data
	$(PYTHON) train_resnet_3ch.py

train_yamnet:
	$(PYTHON) train_yamnet_transfer.py

train_all: train_resnet train_yamnet


# ====== Embedding Extraction ======
extract_resnet:
	$(PYTHON) extract_resnet_embeddings.py

extract_yamnet:
	$(PYTHON) extract_yamnet_embeddings.py

extract_all: extract_resnet extract_yamnet


# ====== Visualizations ======
viz_resnet:
	$(PYTHON) visualize_resnet_embeddings.py

viz_yamnet:
	$(PYTHON) visualize_embeddings.py

viz_waveform_mel: data
	$(PYTHON) visualize_waveform_mel.py

viz_all: viz_resnet viz_yamnet viz_waveform_mel


# ====== Run Tests (pytest) ======
test:
	pytest -q


# ====== Clean (optional) ======
clean:
	rm -rf __pycache__ */__pycache__
	rm -rf $(VIS_DIR)/*.png $(VIS_DIR)/*.html
	rm -rf $*.pth
	rm -f *embeddings.npz


# ====== Full Pipeline ======
# This runs everything needed for final report reproduction
full:
	make install
	make data
	make train_all
	make extract_all
	make viz_all
	make test

.PHONY: install data train_resnet train_yamnet train_all extract_resnet extract_yamnet extract_all viz_resnet viz_yamnet viz_waveform_mel viz_all test clean full
