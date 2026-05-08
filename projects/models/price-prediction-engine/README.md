# Price Prediction Engine

## Overview
Building ML baselines with XGBoost and custom neural networks for price forecasting.

## Concepts
- Feature engineering and preprocessing
- XGBoost baseline models
- Custom Neural Networks (PyTorch)
- Hyperparameter tuning
- Cross-validation and evaluation
- Time series considerations
- Model comparison and analysis

## Project Structure
```
08-price-prediction-engine/
├── src/
│   ├── main.py                  # Training pipeline
│   ├── data_loader.py           # Data ingestion
│   ├── features.py              # Feature engineering
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── neural_network.py
│   │   └── ensemble.py
│   ├── evaluation.py            # Metrics and evaluation
│   └── inference.py             # Prediction API
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── checkpoints/
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

## Status
`[ ] Not Started`

## Stack
- Python 3.10+
- XGBoost
- PyTorch
- Scikit-learn
- Pandas, NumPy
