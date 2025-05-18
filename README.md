# Machine Learning Projects Collection

This repository contains a collection of modular neural network projects for experimentation, education, and prototyping. It includes practical models for structured data tasks, transformer components built from scratch, and experiments with positional embeddings.

---

## 📁 Project Overview

### 🏠 House Prices Neural Network
> _Folder:_ `House Prices Neural Network`

A neural network designed to predict house prices based on various input features. This is a classic regression task useful for learning preprocessing, normalization, and model evaluation.

---

### 🚦 Traffic Volume Neural Network
> _Folder:_ `Traffic Volume Neural Network`

A pipeline for predicting traffic volume using time-series and categorical features. Includes:
- `clean_data.py`: Preprocessing and feature engineering.
- `eval_metrics`: Evaluation functions for model performance.
- `predictions`: Logic for generating and visualizing predictions.
- `train.py`: Model training script.

---

### 🔁 Rotary Positional Embeddings (RoPE)
> _Folder:_ `Rotary Positional Embeddings`

Implements the RoPE mechanism, useful for Transformer architectures where relative positioning is crucial.
- `RoPE.py`: Core implementation.
- `RoPE output on dummy data`: Sample usage with dummy inputs.

---

### ⚙️ Transformer From Scratch
> _Folder:_ `Transformer From Scratch`

A complete, minimal Transformer implementation built entirely from scratch in Python. Great for learning internals and customizing behavior.

Modules include:
- `decoder_block.py`: Transformer decoder architecture.
- `feedforward_mlp.py`: MLP block used within Transformer layers.
- `multi_head_attention.py`: Custom multi-head attention implementation.
- `layer_norm.py`: Layer normalization module.
- `sinusoidal_positional_encodings.py`: Classic positional encoding.
- `imports_logging.py`: Centralized imports and logging helpers.

---

## 📄 License

This project is licensed under the terms of the `LICENSE` file included in the repository.

---

## 🤝 Contributions

Feel free to fork this repo, make improvements, and open pull requests. All contributions that improve learning and clarity are welcome!

