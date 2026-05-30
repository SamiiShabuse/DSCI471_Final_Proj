"""
Fashion multimodal retrieval — shared source package.

Module layout:
  paths.py              Project directories and artifact paths
  config.py             Hyperparameters and constants
  captions.py           Caption and query text builders
  prepare_data.py       Data preprocessing CLI
  download_kaggle_data.py   Dataset download CLI
  model.py              Dual-encoder architecture and encoding
  baseline_keyword.py   TF-IDF keyword search
  search.py             Load models and run retrieval
  metrics.py            Top-K, MRR, Precision@K
  train.py              Training CLI
  evaluate.py           Evaluation CLI
"""
