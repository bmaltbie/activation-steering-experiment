# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based machine learning project that uses the Qwen3-8B model from Hugging Face Transformers. The project is designed to load and initialize a large language model for inference or fine-tuning tasks.

## Development Environment

- **Python Version**: 3.13.1
- **Virtual Environment**: Located in `.venv/` directory
- **Key Dependencies**: transformers, torch, huggingface-hub, safetensors, numpy

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt  # if requirements.txt exists
# Or install specific packages as needed
```

### Running the Application
```bash
# Activate environment and run main script
source .venv/bin/activate && python main.py
```

### Development Tasks
```bash
# Check installed packages
source .venv/bin/activate && pip list

# Update pip (recommended)
source .venv/bin/activate && pip install --upgrade pip
```

## Code Architecture

The project currently consists of a single main module:

- **main.py**: Contains model initialization code for the Qwen3-8B model
  - Loads tokenizer and model using HuggingFace transformers
  - Configured with automatic torch dtype and device mapping
  - Model: "Qwen/Qwen3-8B" (8-billion parameter language model)

## Key Technical Details

- **Model Loading**: Uses `AutoModelForCausalLM` and `AutoTokenizer` from transformers
- **Hardware Optimization**: Configured with `device_map="auto"` for automatic GPU/CPU allocation
- **Memory Management**: Uses `torch_dtype="auto"` for optimal memory usage

## Project Structure

```
├── .venv/          # Python virtual environment
├── .idea/          # JetBrains IDE configuration
├── .git/           # Git repository
└── main.py         # Main application entry point
```

## Notes for Development

- The project is in early stages with minimal structure
- No testing framework is currently configured
- No build system or dependency management files (requirements.txt, setup.py) are present
- Virtual environment should always be activated before running Python commands
- The model requires significant computational resources due to its 8B parameter size