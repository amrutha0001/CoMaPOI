# Single-Agent Inference Module Documentation

## Overview

The Single-Agent Inference Module (`inference_ori_new.py`) is a component of the CoMaPOI (Collaborative Multi-agent POI Prediction) system. This module implements a single-agent approach for predicting the next Point of Interest (POI) a user will visit. It uses either a local vLLM API or OpenAI API to generate predictions.

## Architecture

The script is structured around a modular, object-oriented architecture with the following components:

1. **Utility Functions**: Standalone functions for setting random seeds
2. **OpenAIClient Class**: Handles API client creation and configuration
3. **POIPredictor Class**: Manages prediction generation and processing
4. **InferenceProcessor Class**: Orchestrates the entire inference process
5. **Main Function**: Entry point that parses arguments and initiates the process

## Key Components

### Utility Functions

- **set_random_seed()**: Sets random seeds for reproducibility across all random number generators

### OpenAIClient Class

- **create_client()**: Creates an OpenAI client configured for either OpenAI API or local vLLM

### POIPredictor Class

- **predict_single_sample()**: Predicts POIs for a single sample with retry mechanism
- **process_prediction()**: Processes the generated text to extract and clean predicted POIs

### InferenceProcessor Class

- **__init__()**: Initializes the processor with command line arguments
- **run_inference()**: Runs the inference process using multiple workers

## Inference Workflow

The single-agent inference process follows these steps:

1. **Argument Parsing**: Parse command-line arguments to configure the inference process
2. **Data Loading**: Load samples from the specified dataset
3. **Prompt Creation**: Create prompts for each sample based on the specified format
4. **Parallel Processing**:
   - Create a pool of worker processes
   - Submit prediction tasks for each sample
   - Process results as they complete
5. **Result Processing**:
   - Extract and clean predicted POIs
   - Save interim results at specified intervals
   - Evaluate performance
6. **Final Evaluation**: Evaluate the final predictions and save metrics

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_samples` | int | 0 | Number of samples (0 for all) |
| `--dataset` | str | 'nyc' | Dataset to use ('nyc', 'tky', 'ca') |
| `--data_path` | str | '/home/ZhongLin/SMAC/dataset_all' | Path to dataset |
| `--top_k` | int | 10 | Top K predictions |
| `--api_type` | str | "vllm" | API type |
| `--max_item` | int | 4091 | Maximum POI ID value |
| `--start_point` | int | 0 | Starting point |
| `--test_interval` | int | 500 | Test interval for saving and evaluating |
| `--mode` | str | 'test' | Test mode |
| `--save_name` | str | 'N1' | Save ID |
| `--batch_size` | int | 1 | Number of concurrent processes |
| `--model` | str | 'qwen2.5:7b' | Model to use |
| `--prompt_format` | str | 'json' | Prompt format |
| `--openai` | bool | False | Use OpenAI API instead of local vLLM |
| `--alpaca` | bool | False | Use Alpaca format |
| `--port` | int | 7862 | Port for local vLLM API |
| `--max_tokens` | int | 256 | Maximum tokens for generation |
| `--op_str` | str | 'none' | Operation string for output directory naming |

## Usage Examples

### Basic Usage with Local vLLM

```bash
python inference_ori_new.py --dataset nyc --prompt_format json --model llama3.1-8b-instruct --batch_size 16
```

### Using OpenAI API

```bash
python inference_ori_new.py --dataset nyc --prompt_format json --model gpt-3.5-turbo --batch_size 16 --openai
```

### Specifying Number of Samples

```bash
python inference_ori_new.py --dataset nyc --prompt_format json --model llama3.1-8b-instruct --batch_size 16 --num_samples 100
```

### Using Different Dataset

```bash
python inference_ori_new.py --dataset tky --prompt_format json --model llama3.1-8b-instruct --batch_size 16
```

### Adjusting Batch Size for Performance

```bash
python inference_ori_new.py --dataset nyc --prompt_format json --model llama3.1-8b-instruct --batch_size 32
```

### Using Different Port for vLLM API

```bash
python inference_ori_new.py --dataset nyc --prompt_format json --model llama3.1-8b-instruct --batch_size 16 --port 7863
```

### Using Alpaca Format

```bash
python inference_ori_new.py --dataset nyc --prompt_format json --model llama3.1-8b-instruct --batch_size 16 --alpaca
```

## Output Files

The script generates several output files:

1. **Prediction Results**: `results/{op_str}/{dataset}/{save_name}/poi_predictions.json`
2. **Interim Results**: `results/{op_str}/{dataset}/{save_name}/interim_poi_predictions_{n}.json`
3. **Evaluation Metrics**: 
   - `results/{op_str}/{dataset}/{save_name}/metrics.txt`
   - `results/{op_str}/{dataset}/{save_name}/metrics.csv`

## Prompt Formats

The script supports different prompt formats:

1. **JSON Format**: Structured JSON format for prompts
2. **Original Format**: Plain text format for prompts
3. **Alpaca Format**: Format compatible with Alpaca-style models

## Requirements

- Python 3.8+
- OpenAI Python client
- NumPy
- PyTorch
- tqdm
- concurrent.futures

## Cross-Platform Execution

To run this script on a Mac while using a vLLM API deployed on a Linux server:

1. Ensure the vLLM API is accessible from your Mac (check network connectivity)
2. Update the port in the command line arguments to match your server configuration:
   ```bash
   python inference_ori_new.py --dataset nyc --port 7862 --model llama3.1-8b-instruct
   ```
3. Install the required Python packages on your Mac
4. Run the script as described in the usage examples

## Troubleshooting

- **API Connection Issues**: Check that the vLLM server is running and accessible
- **Empty Predictions**: Check the raw response for formatting issues
- **Performance Issues**: Increase batch size for faster processing
- **Memory Issues**: Reduce batch size if you encounter memory problems
