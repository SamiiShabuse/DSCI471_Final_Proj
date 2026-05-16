# Multimodal Deep Learning Search Engine for E-commerce Fashion

This repository contains the DSCI471 final project: a multimodal (image + text) dual-encoder retrieval system for fashion product search. The project trains a joint embedding space so that product images and natural-language descriptions can be retrieved with semantic similarity rather than exact keyword matches.

Key ideas:
- Dual-encoder architecture (image encoder + text encoder)
- Contrastive training to align matching image-text pairs
- Retrieval by encoding queries and searching product embeddings with cosine similarity

Repository structure
- `data/` — raw and processed datasets. See `data/raw/fashion-dataset` for the original Kaggle files.
- `notebooks/` — exploratory analysis and demo notebooks (e.g., `01_explore_data.ipynb`).
- `src/` — scripts for downloading data, preparing data, and baseline code (`baseline_keyword.py`, `download_kaggle_data.py`, `prepare_data.py`).
- `docs/` — project proposal and documentation.

Project flow
- Place raw data (JSON + images) under `data/raw/fashion-dataset`.
- Run `python src/prepare_data.py` to extract text, normalize images, and produce `data/processed/products.csv` used by notebooks and training.
- (Modeling) Use the processed CSV to train a dual-encoder: compute image and text embeddings, train with contrastive loss, and save model checkpoints and computed product embeddings for fast retrieval.
- Evaluate and demo using the notebooks in `notebooks/` which load the processed CSV and saved embeddings to visualize retrieval results.


Quickstart (Windows)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset (Kaggle)

This project uses the Fashion Product Images dataset (Kaggle). You can download it using the included helper or manually place the files under `data/raw/fashion-dataset`:

```bash
python src/download_kaggle_data.py
# or manually: place 'styles.csv', 'images.csv', and folders under data/raw/fashion-dataset
```

4. Prepare the data

Preprocess raw JSON, images, and metadata into a compact `data/processed/` CSV used by training and notebooks:

```bash
python src/prepare_data.py
```

5. Explore and run demos

Open the notebooks in `notebooks/` to explore data and run a demo retrieval:

```bash
jupyter lab notebooks/
```

Notes on training and modeling
- Model code and training examples live in the `src/` folder and notebooks. The intended approach is a pretrained CNN (ResNet/EfficientNet) for images and a Transformer (or lightweight text encoder) for text, trained with contrastive loss.
- Training can be expensive — use Google Colab Pro or a GPU-enabled machine for full experiments.

Data description
- `data/raw/fashion-dataset/styles/` — per-product JSON records (attributes, descriptions).
- `data/raw/fashion-dataset/images/` — product images (images/{id}.jpg).
- `data/processed/products.csv` — compact table of product id, text, and processed image paths (created by `src/prepare_data.py`).

How to show this on your resume / talking points
- Implemented a multimodal retrieval system: trained dual-encoder to align images and text for semantic search.
- Used TensorFlow/Keras and pretrained CNN backbones for efficient feature extraction.
- Built data pipeline to convert raw product JSON and images into trainable image-text pairs.
- Evaluated retrieval performance with Top-K accuracy, MRR, and Precision@K; created visualizations in notebooks.

Troubleshooting & tips
- If dataset download fails, manually obtain files from Kaggle and place them under `data/raw/fashion-dataset`.
- For Windows path issues, ensure the virtual environment is activated before running scripts.
- If you run out of GPU memory, reduce batch size or use a smaller backbone.

Credits and license
This project was developed for the DSCI471 course. See `docs/DSCI471 Project Proposal.md` for the project plan and timeline.


