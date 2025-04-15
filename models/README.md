# Models Directory

This directory contains model files for the CoMaPOI project.

## Model Structure

The models directory should be organized as follows:

```
models/
├── bce_embedding/           # BCE embedding model
├── llama3.1-8b-instruct/    # Base LLaMA 3.1 model
├── qwen2.5-7b-instruct/     # Base Qwen 2.5 model
└── fine-tuned/              # Fine-tuned models
    ├── nyc/                 # NYC dataset fine-tuned models
    │   ├── agent1/          # Profiler agent
    │   ├── agent2/          # Forecaster agent
    │   └── agent3/          # Final_Predictor agent
    ├── tky/                 # Tokyo dataset fine-tuned models
    │   ├── agent1/          # Profiler agent
    │   ├── agent2/          # Forecaster agent
    │   └── agent3/          # Final_Predictor agent
    └── ca/                  # California dataset fine-tuned models
        ├── agent1/          # Profiler agent
        ├── agent2/          # Forecaster agent
        └── agent3/          # Final_Predictor agent
```

## Model Download

You can download the base models from their respective sources:

- LLaMA 3.1: [Meta AI](https://llama.meta.com/)
- Qwen 2.5: [Alibaba Cloud](https://qianwen.aliyun.com/)
- BCE Embedding: [BCE-embedding-base](https://huggingface.co/maidalun1020/bce-embedding-base_v1)

## Fine-tuned Models

The fine-tuned models are created using the `finetune_sft_new.py` script. You can fine-tune your own models using the following command:

```bash
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type agent1 --batch_size 16 --max_steps 200 --bf16
```

## Using Models with vLLM

To deploy the models with vLLM, use the following command:

```bash
python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port 7862 \
    --model models/llama3.1-8b-instruct \
    --served-model-name llama3.1-8b \
    --tensor-parallel-size 1 \
    --dtype=auto \
    --enable-lora \
    --gpu-memory-utilization 0.9 \
    --disable-log-stats \
    --disable-log-requests \
    --max_loras 3 \
    --lora-modules agent1=models/fine-tuned/nyc/agent1 \
    agent2=models/fine-tuned/nyc/agent2 \
    agent3=models/fine-tuned/nyc/agent3
```
