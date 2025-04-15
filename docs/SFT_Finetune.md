# Supervised Fine-Tuning (SFT) Module Documentation

## Overview

The Supervised Fine-Tuning (SFT) Module (`finetune_sft_new.py`) is a critical component of the CoMaPOI (Collaborative Multi-agent POI Prediction) system. This module performs parameter-efficient fine-tuning of large language models (LLMs) using the LoRA (Low-Rank Adaptation) technique to adapt them for specific agent roles within the CoMaPOI framework.

## Architecture

The script is structured around a modular, object-oriented architecture with the following components:

1. **Helper Functions**: Standalone utility functions for text processing and token counting
2. **DataProcessor Class**: Handles data loading, processing, and preparation
3. **ModelTrainer Class**: Manages model initialization and training
4. **Main Function**: Entry point that parses arguments and orchestrates the process

## Key Components

### Helper Functions

- **prepare_sample_text()**: Formats dataset samples into text suitable for training
- **count_tokens()**: Counts the number of tokens in a sample
- **chars_token_ratio()**: Estimates the average number of characters per token in the dataset
- **print_trainable_parameters()**: Prints the number of trainable parameters in the model

### DataProcessor Class

- **process_and_split_jsonl()**: Processes JSONL files and splits data into training and test sets
- **merge_agent_files()**: Merges data from multiple agent files
- **check_and_process_files()**: Checks if processed files exist and creates them if needed
- **create_datasets()**: Creates and prepares datasets for training

### ModelTrainer Class

- **run_training()**: Initializes models and runs the training process

## Training Workflow

The fine-tuning process follows these steps:

1. **Argument Parsing**: Parse command-line arguments to configure the training process
2. **Data Preparation**: 
   - Process and clean training data
   - Merge data from different agents if needed
   - Split data into training and test sets
3. **Model Initialization**:
   - Load the base model with quantization
   - Apply LoRA adapters to make training efficient
   - Configure tokenizer
4. **Training**:
   - Set up SFTTrainer with appropriate parameters
   - Train the model for the specified number of steps
   - Save checkpoints at regular intervals
5. **Model Saving**: Save the final fine-tuned model

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | "/home/ZhongLin/LLM/" | Path to pretrained model |
| `--model` | str | "qwen2.5-7b-instruct" | Pretrained model name |
| `--dataset` | str | "nyc" | Dataset name (nyc, tky, ca) |
| `--data_path` | str | "" | Training data path |
| `--seq_length` | int | 2048 | Maximum sequence length |
| `--max_steps` | int | 200 | Maximum training steps |
| `--num_samples` | int | 0 | Number of samples (0 for all) |
| `--batch_size` | int | 4 | Batch size per device |
| `--gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `--learning_rate` | float | 1e-4 | Learning rate |
| `--fp16` | bool | False | Use FP16 training |
| `--bf16` | bool | False | Use BF16 training |
| `--gradient_checkpointing` | bool | False | Use gradient checkpointing |
| `--output_dir` | str | "output" | Output directory |
| `--save_freq` | int | 40 | Model saving frequency |
| `--type` | str | 'merged' | Training type (merged, agent1, agent2, agent3) |
| `--unsloth` | bool | False | Use Unsloth acceleration |

## Usage Examples

### Basic Usage

```bash
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type merged --batch_size 16 --max_steps 200
```

### Fine-tuning with BF16 Precision

```bash
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type merged --batch_size 16 --max_steps 200 --bf16
```

### Fine-tuning with Unsloth Acceleration

```bash
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type merged --batch_size 16 --max_steps 200 --unsloth --bf16
```

### Fine-tuning Individual Agents

```bash
# Fine-tune Profiler (Agent 1)
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type agent1 --batch_size 16 --max_steps 200 --bf16

# Fine-tune Forecaster (Agent 2)
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type agent2 --batch_size 16 --max_steps 200 --bf16

# Fine-tune Final_Predictor (Agent 3)
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type agent3 --batch_size 16 --max_steps 200 --bf16
```

### Fine-tuning with Different Learning Rates

```bash
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type merged --batch_size 16 --max_steps 200 --learning_rate 5e-5 --bf16
```

## Output Files

The script generates several output files:

1. **Processed Data Files**:
   - `agent{1,2,3}_train_samples_all.jsonl`: Processed training data for each agent
   - `agent{1,2,3}_train_samples_100.jsonl`: Processed test data for each agent
   - `total_agent_train_samples.jsonl`: Merged data from all agents
   - `cleaned_total_agent_train_samples_all.jsonl`: Cleaned merged training data

2. **Model Files**:
   - `finetune/results/{op_str}/sft-{dataset}/{save_name}/`: Directory containing model checkpoints
   - `adapter_model.bin`: LoRA adapter weights
   - `adapter_config.json`: LoRA adapter configuration

## Fine-tuning Approaches

The script supports different fine-tuning approaches:

1. **Individual Agent Fine-tuning**:
   - `agent1`: Fine-tune for the Profiler agent
   - `agent2`: Fine-tune for the Forecaster agent
   - `agent3`: Fine-tune for the Final_Predictor agent

2. **Merged Fine-tuning**:
   - `merged`: Fine-tune a single model using data from all agents

## Optimization Techniques

The script implements several optimization techniques:

1. **Parameter-Efficient Fine-Tuning (PEFT)**:
   - Uses LoRA to fine-tune only a small subset of parameters
   - Reduces memory requirements and training time

2. **Quantization**:
   - Uses 4-bit quantization via BitsAndBytes
   - Enables fine-tuning of larger models on limited hardware

3. **Unsloth Acceleration**:
   - Optional acceleration for single-GPU training
   - Optimizes memory usage and computation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- TRL (Transformer Reinforcement Learning)
- PEFT (Parameter-Efficient Fine-Tuning)
- Unsloth (optional)

## Cross-Platform Execution

To run this script on a Mac while using models deployed on a Linux server:

1. Modify the model paths to point to your local model directory
2. Adjust batch sizes and precision settings based on your hardware
3. Consider using smaller models or more aggressive quantization for Mac hardware

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size, enable gradient checkpointing, or use more aggressive quantization
- **Slow Training**: Enable Unsloth acceleration, reduce sequence length, or use a smaller model
- **Poor Results**: Try different learning rates, increase training steps, or check data quality
- **Data Format Issues**: Ensure your data follows the expected format with 'user' and 'assistant' messages
