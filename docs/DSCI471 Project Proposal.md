# Project Title: Multimodal Deep Learning Search Engine for E-commerce Fashion

# Team Members: Richardson Chhin, Samii Shabuse

# Research Questions and Motivations:
Can a dual-encoder deep learning model, trained on paired fashion images and text descriptions, outperform traditional keyword-based e-commerce search? Existing search systems rely heavily on exact keyword matching, which fails when users describe products in natural language (Ex: "a flowy floral dress with cap sleeves"). This project aims to learn a shared embedding space aligning visual and textual representations, enabling semantic search that returns contextually relevant products without requiring exact keyword matches. Improving product discovery has real-world impact, including higher conversion rates, reduced bounce rates, and fewer product returns in online retail.

# Data Collection:
We will use the Fashion Product Images Dataset (Aggarwal, Kaggle), sourced from a real-world e-commerce catalog. It contains approximately 44,400 professionally shot product images, manually entered label attributes (Ex: gender, category, color, season, usage), and descriptive text commenting on product characteristics. Each product is identified by a unique ID linking its image (images/{id}.jpg), structured metadata (styles.csv), and full JSON record (styles/{id}.json). We will construct image-text pairs by combining product names, label attributes, and descriptive text into a unified text representation matched to each product image. Preprocessing includes image resizing/normalization, text tokenization, and train/validation/test splits.

# Methodology:
We will implement and evaluate a dual-encoder architecture using TensorFlow and Keras. A pretrained CNN (Ex: ResNet or EfficientNet) will encode images into feature embeddings, while a Transformer-based text encoder (or LSTM baseline) will encode product descriptions into the same embedding space. The model will be trained using contrastive learning to bring matching pairs closer and push non-matching pairs apart. At inference, user queries are encoded and matched against precomputed image embeddings using cosine similarity for retrieval. To validate performance, we will compare against a baseline keyword-based search model. Evaluation metrics will include Top-K accuracy (Top-1, Top-5), Mean Reciprocal Rank (MRR), and Precision@K. Potential challenges include noisy text descriptions and computational constraints, which will be addressed through preprocessing and the use of pretrained models.

# Timeline:
Week 6: Finalize project scope, submit proposal, dataset preparation, preprocessing
Week 7–8: Develop and test image and text encoders. Establish a training pipeline
Week 9: Combine encoders into the joint architecture, train with contrastive loss, tune hyperparameters, and evaluate retrieval accuracy
Week 10: Generate results, visualizations. Deliver an end-of-term presentation
Week 11 / Finals Week: Complete final report with detailed discussion of deep learning methods, results, and limitations.

# Resources:
Software: Python, Jupyter Notebook, TensorFlow, Keras, NumPy, scikit-learn, GitHub
Hardware: Google Colab Pro or a GPU-accelerated local machine (CUDA or Apple Metal)
