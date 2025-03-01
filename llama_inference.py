#!/usr/bin/env python3
"""
Llama 3.2 3B Inference Script
This script demonstrates how to load and use the Meta-Llama/Llama-3.2-3B model from Hugging Face.
"""

import os
import argparse
import yaml
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_config(config_path="cfg.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default settings")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_model(model_name, device="auto", precision="bf16", cache_dir=None):
    """
    Load the model and tokenizer.
    
    Args:
        model_name: Name or path of the model on Hugging Face
        device: Device to load the model on ("auto", "cuda", "cpu")
        precision: Model precision (bf16, fp16, fp32)
        cache_dir: Directory to cache downloaded models
        
    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Common kwargs for both tokenizer and model
    common_kwargs = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        common_kwargs["cache_dir"] = cache_dir
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
    
    # Set torch dtype based on precision
    if precision == "bf16" and torch.cuda.is_available():
        print("Using GPU with BF16 precision")
        torch_dtype = torch.bfloat16
    elif precision == "fp16" and torch.cuda.is_available():
        print("Using GPU with FP16 precision")
        torch_dtype = torch.float16
    else:
        print("Using FP32 precision")
        torch_dtype = torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch_dtype,
        **common_kwargs
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, **gen_kwargs):
    """
    Generate a response from the model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: The input prompt
        gen_kwargs: Generation parameters
        
    Returns:
        Generated text
    """
    # Format the prompt according to the model's expected format
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|assistant|>")[-1].strip()
    return assistant_response

def main():
    parser = argparse.ArgumentParser(description="Run inference with Llama 3.2 3B model")
    parser.add_argument("--config", type=str, default="cfg.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility if provided
    if "system" in config and "seed" in config["system"]:
        random.seed(config["system"]["seed"])
    
    # Check if HUGGING_FACE_TOKEN is set
    if "HUGGING_FACE_TOKEN" not in os.environ:
        print("Warning: HUGGING_FACE_TOKEN environment variable not set.")
        print("You may need to set it if the model requires authentication:")
        print("export HUGGING_FACE_TOKEN=your_token_here")
    
    # Get model settings from config
    model_config = config.get("model", {})
    model_name = model_config.get("name", "meta-llama/Llama-3.2-3B-Instruct")
    device = model_config.get("device", "auto")
    precision = model_config.get("precision", "bf16")
    
    # Get system settings
    system_config = config.get("system", {})
    cache_dir = system_config.get("cache_dir", None)
    
    # Load model and tokenizer
    model, tokenizer = setup_model(
        model_name, 
        device=device, 
        precision=precision,
        cache_dir=cache_dir
    )
    
    # Get generation settings
    gen_config = config.get("generation", {})
    gen_kwargs = {
        "max_new_tokens": gen_config.get("max_new_tokens", 512),
        "temperature": gen_config.get("temperature", 0.7),
        "top_p": gen_config.get("top_p", 0.9),
        "top_k": gen_config.get("top_k", 50),
    }
    
    # Get prompt
    for prompt in config["prompts"]:
        # Generate response
        print("\nPrompt:", prompt, "\n")
        print("\nGenerating response...\n")
        response = generate_response(
            model, 
            tokenizer, 
            prompt,
            **gen_kwargs
        )
        
        print("Response:", response)
        print("\n")

if __name__ == "__main__":
    main() 