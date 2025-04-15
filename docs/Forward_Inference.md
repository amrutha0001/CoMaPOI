# Forward Inference Module Documentation

## Overview

The Forward Inference Module (`inference_forward_new.py`) is a core component of the CoMaPOI (Collaborative Multi-agent POI Prediction) system. This module implements a multi-agent approach for predicting the next Point of Interest (POI) a user will visit. It coordinates three specialized agents (Profiler, Forecaster, and Final_Predictor) that work together to analyze user profiles, mobility patterns, and contextual information to make accurate predictions.

## Architecture

The script is structured around a modular architecture with the following components:

1. **Helper Functions**: Standalone utility functions for data processing, POI handling, and text extraction
2. **Agent Initialization**: Functions to initialize and configure the three specialized agents
3. **Agent Step Functions**: Functions that implement the specific steps for each agent
4. **Prediction Functions**: Functions that handle the prediction process for individual samples
5. **ForwardInferenceProcessor Class**: Main class that orchestrates the entire prediction process
6. **Main Function**: Entry point that parses arguments and initiates the process

## Key Components

### Helper Functions

- **get_num_tokens()**: Counts the number of tokens in a prompt
- **extract_and_clean_poi()**: Extracts and cleans predicted POIs from text
- **merge_valid_pois()**: Merges existing POIs with new POIs, ensuring no duplicates
- **parse_reasoning_path()**: Parses reasoning path from a JSON file
- **check_extra_information()**: Extracts information from JSON strings
- **get_candidate_poi_lists()**: Processes candidate POI lists from both agents

### Agent Initialization

- **init_agents()**: Initializes the three specialized agents (Profiler, Forecaster, Final_Predictor)
- **React_get_profile_information()**: Gets profile information for a specific user
- **process_single_profile()**: Processes a single user profile in a separate process
- **React_process_and_save_profiles()**: Processes and saves user profiles in parallel

### Agent Step Functions

- **profiler_steps()**: Executes the Profiler agent steps to generate long-term profile and candidate POIs
- **forecaster_steps()**: Executes the Forecaster agent steps to generate short-term pattern and refined candidates
- **final_prediction_steps()**: Executes the Final_Predictor agent steps to generate the final prediction
- **validate_and_retry_sample()**: Validates POI IDs and retries if invalid

### Prediction Functions

- **single_predict_save()**: Predicts POIs for a single sample using saved reasoning paths
- **single_predict()**: Predicts POIs for a single sample using the full agent pipeline

### ForwardInferenceProcessor Class

- **__init__()**: Initializes the processor with command line arguments
- **parallel_predict()**: Runs parallel prediction using multiple processes

## Multi-Agent Workflow

The forward inference process follows a collaborative multi-agent workflow:

1. **Profiler Agent (Agent 1)**:
   - Analyzes the user's historical trajectory data
   - Generates a long-term user profile
   - Proposes candidate POIs based on historical patterns

2. **Forecaster Agent (Agent 2)**:
   - Analyzes the user's recent trajectory data
   - Identifies short-term mobility patterns
   - Refines candidate POIs using RAG (Retrieval-Augmented Generation)

3. **Final_Predictor Agent (Agent 3)**:
   - Combines insights from both Profiler and Forecaster
   - Considers both long-term preferences and short-term patterns
   - Makes the final prediction of the next POI

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_samples` | int | 0 | Number of samples to process (0 uses dataset default) |
| `--dataset` | str | 'nyc' | Dataset to use ('nyc', 'tky', 'ca') |
| `--top_k` | int | 10 | Top K predictions |
| `--model` | str | 'qwen2.5-1.5b-instruct' | Base model to use |
| `--api_type` | str | "gpt" | API type for model |
| `--max_item` | int | 5091 | Maximum POI ID value |
| `--max_retry` | int | 1 | Maximum number of retries |
| `--start_point` | int | 0 | Starting point for processing |
| `--test_interval` | int | 200 | Test interval for saving and evaluating |
| `--batch_size` | int | 8 | Number of concurrent processes |
| `--mode` | str | 'test' | Mode (train/test) |
| `--save_name` | str | 'N1' | Save ID (N1-N...; T1-T...; C1-C...) |
| `--agent1_api` | str | 'agent1' | API name for Agent 1 (Profiler) |
| `--agent2_api` | str | 'agent2' | API name for Agent 2 (Forecaster) |
| `--agent3_api` | str | 'agent3' | API name for Agent 3 (Final_Predictor) |
| `--num_candidate` | int | 25 | Number of candidate POIs |
| `--store_save_name` | bool | False | Manually provide storage name |
| `--port` | int | 7862 | Port for API server |
| `--agent1_max_tokens` | int | 256 | Max tokens for Agent 1 |
| `--agent2_max_tokens` | int | 256 | Max tokens for Agent 2 |
| `--agent3_max_tokens` | int | 256 | Max tokens for Agent 3 |
| `--sub_file` | str | 'ablation' | Sub-directory for results |
| `--load_pf_output` | bool | False | Load pre-generated profiles |
| `--saved_results_path` | str | 'none' | Path to saved results |
| `--op_str` | str | 'none' | Operation string |
| `--temperature` | float | 0.0 | Temperature for generation |
| `--top_p` | float | 1 | Top-p for generation |
| `--n` | int | 1 | Number of generations |
| `--prompt_format` | str | "json" | Prompt format |
| `--ab_type` | str | "none" | Ablation type |
| `--seed` | int | 0 | Random seed |

## Usage Examples

### Basic Usage

```bash
python inference_forward_new.py --dataset nyc --model llama3.1-8b-instruct --agent1_api agent1 --agent2_api agent2 --agent3_api agent3
```

### Specifying Number of Samples

```bash
python inference_forward_new.py --dataset nyc --model llama3.1-8b-instruct --num_samples 100 --agent1_api agent1 --agent2_api agent2 --agent3_api agent3
```

### Using Different Dataset

```bash
python inference_forward_new.py --dataset tky --model llama3.1-8b-instruct --agent1_api agent1-tky --agent2_api agent2-tky --agent3_api agent3-tky
```

### Adjusting Batch Size for Performance

```bash
python inference_forward_new.py --dataset nyc --model llama3.1-8b-instruct --batch_size 16 --agent1_api agent1 --agent2_api agent2 --agent3_api agent3
```

### Running Ablation Studies

```bash
python inference_forward_new.py --dataset nyc --model llama3.1-8b-instruct --ab_type profiler --agent1_api agent1 --agent2_api agent2 --agent3_api agent3
```

### Loading Pre-generated Profiles

```bash
python inference_forward_new.py --dataset nyc --model llama3.1-8b-instruct --load_pf_output --saved_results_path results/nyc/profiles/poi_predictions.json
```

## Output Files

The script generates several output files:

1. **User Profiles**: `dataset_all/{dataset}/{dataset}_historical_summary.jsonl`
2. **Prediction Results**: `results/{dataset}/{save_name}/poi_predictions.json`
3. **Interim Results**: `results/{dataset}/{save_name}/interim_poi_predictions_{n}.json`
4. **Evaluation Metrics**: 
   - `results/{dataset}/{save_name}/metrics.txt`
   - `results/{dataset}/{save_name}/metrics.csv`

## Ablation Studies

The script supports ablation studies to evaluate the contribution of each agent:

- **profiler**: Disables the Profiler agent (Agent 1)
- **forecaster**: Disables the Forecaster agent (Agent 2)
- **candidate**: Disables candidate POI lists from both agents

## Requirements

- Python 3.8+
- AgentScope
- Transformers
- tqdm
- concurrent.futures

## Notes

- The script uses a multiprocessing approach to parallelize the prediction process
- Each worker process initializes its own agents to avoid serialization issues
- The script is designed to work with a local VLLM API server
- Progress is displayed using a green tqdm progress bar

## Cross-Platform Execution

To run this script on a Mac while using a VLLM API deployed on a Linux server:

1. Ensure the VLLM API is accessible from your Mac (check network connectivity)
2. Update the port in the command line arguments to match your server configuration:
   ```bash
   python inference_forward_new.py --dataset nyc --port 7862 --agent1_api agent1 --agent2_api agent2 --agent3_api agent3
   ```
3. Install the required Python packages on your Mac
4. Run the script as described in the usage examples

## Troubleshooting

- If you encounter serialization errors, ensure that the worker functions are properly isolated
- If the VLLM API is not responding, check that the server is running and accessible
- For memory issues, try reducing the batch size
- If the script is slow, consider increasing the batch size if you have sufficient CPU cores
- If you encounter issues with agent responses, try adjusting the temperature and top_p parameters

## Reproducing Best Performance

To reproduce our best performance (using TKY dataset as an example):

1. Our fine-tuned agent checkpoints are located at:
   - agent1: `CoMaPOI/results/tky/checkpoint-agent1/adapter_model.safetensors`
   - agent2: `CoMaPOI/results/tky/checkpoint-agent2/adapter_model.safetensors`
   - agent3: `CoMaPOI/results/tky/checkpoint-agent3/adapter_model.safetensors`

2. Deploy the three agents with VLLM using:
   ```bash
   CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
       --host 127.0.0.1 \
       --port 7863 \
       --model /home/ZhongLin/LLM/llama3.1-8b-instruct \
       --served-model-name llama3.1-8b\
       --tensor-parallel-size 1 \
       --dtype=auto \
       --enable-lora \
       --gpu-memory-utilization 0.9 \
       --disable-log-stats \
       --disable-log-requests \
       --max_loras 1 \
       --lora-modules agent1=/home/ZhongLin/SMAC/finetune/results/LLAMA3.1-BEST-TKY/Parameters/checkpoint-80-agent1 \
       agent2=/home/ZhongLin/SMAC/finetune/results/LLAMA3.1-BEST-TKY/Parameters/checkpoint-80-agent2 \
       agent3=/home/ZhongLin/SMAC/finetune/results/LLAMA3.1-BEST-TKY/Parameters/checkpoint-80-agent3
   ```

3. Run the inference:
   ```bash
   python inference_forward_new.py --dataset tky --model llama3.1-8b --agent1_api agent1 --agent2_api agent2 --agent3_api agent3 --batch_size 32 
   ```

4. Expected performance:
   - HR@5: 45.83
   - HR@10: 54.26
   - NDCG@5: 34.48
   - NDCG@10: 37.20
   - MRR: 31.82
