import transformers
import torch

MODEL_PIPELINE = None
TOKENIZER = None

def load_models():
    global MODEL_PIPELINE
    global TOKENIZER
    if not MODEL_PIPELINE:
        MODEL_PIPELINE = transformers.pipeline(
            "text-generation",
            model="/models/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
        TOKENIZER = MODEL_PIPELINE.tokenizer
        print("Loaded model.")

