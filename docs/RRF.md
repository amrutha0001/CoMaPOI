# Reverse Inference Module (RRF) Documentation

## Overview

The Reverse Inference Module (`inference_inverse_new.py`) is a key component of the CoMaPOI (Collaborative Multi-agent POI Prediction) system. This module performs inverse inference to generate synthetic training data based on target Points of Interest (POIs). It uses language models to create realistic user profiles, mobility patterns, and POI preferences that can be used to train the forward prediction models.

## Architecture

The script is structured around a modular architecture with the following components:

1. **Helper Functions**: Standalone utility functions for text processing, POI handling, and agent interactions
2. **Worker Function**: A dedicated function for parallel processing that initializes its own agents
3. **InverseInferenceProcessor Class**: Main class that orchestrates the data generation process
4. **Main Function**: Entry point that parses arguments and initiates the process

## Key Components

### Helper Functions

- **init_agents()**: Initializes language model agents for generation tasks
- **generate_by_agent()**: Generates content using an agent
- **extract_and_clean_poi()**: Extracts and cleans predicted POIs from text
- **extract_text()**: Extracts specific fields from JSON strings
- **ensure_label_first()**: Ensures a specific label is the first item in a list
- **complete_candidate_poi_list()**: Completes a candidate POI list to reach a target length
- **process_and_save_profiles()**: Processes and saves user profiles

### Worker Function

- **single_predict_worker()**: Performs prediction for a single sample in a separate process

### InverseInferenceProcessor Class

- **generate_jsonl_files()**: Generates JSONL files for model fine-tuning
- **split_and_save_by_user_info()**: Splits and saves JSONL by user info
- **process_data()**: Processes data from input JSON file and creates JSONL files
- **save_generated_samples()**: Saves generated samples to files
- **run_parallel_predict()**: Runs parallel prediction using multiple processes

## Workflow

1. **Initialization**: Parse command-line arguments and initialize the processor
2. **Data Loading**: Load samples and candidate POIs
3. **Parallel Processing**: Process samples in parallel using multiple workers
4. **Data Generation**: For each sample:
   - Generate historical distribution
   - Generate candidate POIs from profile
   - Generate recent mobility analysis
   - Generate candidate POIs from RAG
   - Generate negative POI list
   - Generate forward prompts
5. **Data Saving**: Save generated data to JSON and JSONL files
6. **Data Processing**: Process data for agent training

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_samples` | int | 0 | Number of samples to process (0 uses dataset default) |
| `--dataset` | str | 'nyc' | Dataset to use ('nyc', 'tky', 'ca') |
| `--top_k` | int | 10 | Top K predictions |
| `--api_type` | str | "qwen2.5:3b" | API type for model |
| `--max_item` | int | 5091 | Maximum POI ID value |
| `--max_retry` | int | 1 | Maximum number of retries |
| `--start_point` | int | 0 | Starting point for processing |
| `--test_interval` | int | 50 | Test interval |
| `--batch_size` | int | 32 | Number of concurrent processes |
| `--mode` | str | 'train' | Mode (train/test) |
| `--save_id` | str | 'N1' | Save ID (N1-N...; T1-T...; C1-C...) |

## Usage Examples

### Basic Usage

```bash
python inference_inverse_new.py --dataset nyc --api_type qwen2.5-7b-instruct --batch_size 32
```

### Specifying Number of Samples

```bash
python inference_inverse_new.py --dataset nyc --api_type qwen2.5-7b-instruct --num_samples 100
```

### Using Different Dataset

```bash
python inference_inverse_new.py --dataset tky --api_type qwen2.5-7b-instruct --batch_size 32
```

### Adjusting Batch Size for Performance

```bash
python inference_inverse_new.py --dataset nyc --api_type qwen2.5-7b-instruct --batch_size 64
```

### Starting from a Specific Point

```bash
python inference_inverse_new.py --dataset nyc --api_type qwen2.5-7b-instruct --start_point 500
```

### Using Test Mode

```bash
python inference_inverse_new.py --dataset nyc --api_type qwen2.5-7b-instruct --mode test
```

## Output Files

The script generates several output files:

1. **User Profiles**: `dataset_all/{dataset}/{dataset}_historical_summary.jsonl`
2. **Generated Information**: `finetune/data/{dataset}/ALL_generated_informations.json`
3. **POI Predictions**: `finetune/data/{dataset}/final_poi_predictions.json`
4. **Fine-tuning Data**: `finetune/data/{dataset}/final_ft_data/`
5. **Agent Training Samples**:
   - `finetune/data/{dataset}/agent1_train_samples.jsonl`
   - `finetune/data/{dataset}/agent2_train_samples.jsonl`
   - `finetune/data/{dataset}/agent3_train_samples.jsonl`

## Requirements

- Python 3.8+
- AgentScope
- Transformers
- tqdm
- concurrent.futures

## Notes

- The script uses a multiprocessing approach to parallelize the data generation process
- Each worker process initializes its own agents to avoid serialization issues
- The script is designed to work with a local VLLM API server running on port 7862
- Progress is displayed using a green tqdm progress bar

## Cross-Platform Execution

To run this script on a Mac while using a VLLM API deployed on a Linux server:

1. Ensure the VLLM API is accessible from your Mac (check network connectivity)
2. Update the base URL in the `init_agents` function to point to your server:
   ```python
   "base_url": "http://your-server-ip:7862/v1"
   ```
3. Install the required Python packages on your Mac
4. Run the script as described in the usage examples

## Troubleshooting

- If you encounter serialization errors, ensure that the `single_predict_worker` function is properly isolated
- If the VLLM API is not responding, check that the server is running and accessible
- For memory issues, try reducing the batch size
- If the script is slow, consider increasing the batch size if you have sufficient CPU cores
