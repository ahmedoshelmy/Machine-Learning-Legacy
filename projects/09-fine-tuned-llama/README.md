# Fine-tuned Llama 3.2

## Overview
Supervised fine-tuning of Llama 3.2 using QLoRA for domain-specific tasks.

## Concepts
- QLoRA efficient fine-tuning
- LoRA adapters and weights
- Custom dataset preparation
- Training loops and validation
- Inference with fine-tuned models
- Model quantization
- Performance evaluation

## Project Structure
```
09-fine-tuned-llama/
├── src/
│   ├── main.py                  # Training entry point
│   ├── data_prep.py             # Dataset preparation
│   ├── training.py              # Training loop
│   ├── inference.py             # Model inference
│   └── utils.py                 # Utilities
├── data/
│   ├── raw/
│   ├── processed/
│   └── validation/
├── models/
│   ├── lora_adapters/
│   └── checkpoints/
├── config/
│   └── training_config.yaml
├── requirements.txt
└── README.md
```

## Status
`[ ] Not Started`

## Stack
- Python 3.10+
- Llama (via Hugging Face Transformers)
- PEFT (LoRA/QLoRA)
- PyTorch
- BitsAndBytes (quantization)
