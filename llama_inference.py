#!/usr/bin/env python3
"""
Llama 3.2 3B Inference Script
This script demonstrates how to load and use the Meta-Llama/Llama-3.2-3B model from Hugging Face.
"""

import os
import argparse
import yaml
import random
import json
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

def generate_response(model, tokenizer, prompt, config=None, **gen_kwargs):
    """
    Generate a response from the model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: The input prompt
        config: Configuration dictionary
        gen_kwargs: Generation parameters
        
    Returns:
        Generated text
    """
    # Get system prompt from config if available, otherwise use default
    default_system_prompt = "You are an AI assistant providing concise, accurate, and helpful responses. Prioritize clarity, efficiency, and ethical engagement. Adapt to user preferences while ensuring safety and objectivity. Structure responses logically, using code, lists, or examples when needed. Avoid speculation, biases, or harmful content. When unsure, clarify or acknowledge limitations. For technical topics, follow best practices and offer multiple perspectives when relevant."
    
    system_prompt = config.get("system_prompt", default_system_prompt) if config else default_system_prompt
    
    # Use a simple format for all models
    formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    # Tokenize the prompt
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **gen_kwargs
            )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        parts = full_response.split("Assistant:")
        if len(parts) > 1:
            assistant_response = parts[-1].strip()
        else:
            # Fallback: return everything after the prompt
            assistant_response = full_response[len(formatted_prompt):].strip()
        
        return assistant_response
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {e}"

def load_jsonl_dataset(file_path):
    """
    Load prompts from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of prompt data
    """
    print(f"Loading dataset from: {file_path}")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} prompts")
    return data

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
    
    # Initialize results list for output
    results = []
    
    # Get dataset settings from config
    dataset_config = config.get("dataset", {})
    dataset_path = dataset_config.get("path")
    num_samples = dataset_config.get("num_samples", 5)
    output_path = dataset_config.get("output")
    
    # Replace {model_name} in output path if present
    if output_path and "{model_name}" in output_path:
        model_name_short = model_name.split("/")[-1]  # Extract just the model name without the org
        output_path = output_path.replace("{model_name}", model_name_short)
    
    # Create output directory if it doesn't exist
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output will be saved to: {output_path}")
    
    # Process prompts from dataset if provided, otherwise use config prompts
    if dataset_path:
        try:
            # Load dataset
            dataset = load_jsonl_dataset(dataset_path)
            
            # Determine number of samples to process
            num_samples = len(dataset) if num_samples == 0 else min(num_samples, len(dataset))
            samples = dataset[:num_samples]
            
            print(f"Processing {num_samples} samples from dataset")
            
            for i, sample in enumerate(samples):
                print(f"\nSample {i+1}/{num_samples}")
                
                # Get the prompt from the prompt_list (we verified each sample has exactly one prompt)
                prompt = sample["prompt_list"][0]
                
                # Make sure prompt is a string before slicing for preview
                prompt_preview = str(prompt)[:100] + "..." if len(str(prompt)) > 100 else str(prompt)
                print(f"\nPrompt: {prompt_preview}\n")
                print("\nGenerating response...\n")
                
                response = generate_response(
                    model, 
                    tokenizer, 
                    prompt,
                    config=config,
                    **gen_kwargs
                )
                
                print("Response:", response)
                print("\n")
                
                # Store result
                result = {
                    "sample_id": i,
                    "prompt": prompt,
                    "response": response,
                    "high_reward_answer": sample.get("high_reward_answer"),
                    "other_answers": sample.get("other_answers")
                }
                results.append(result)
        except FileNotFoundError:
            print(f"Dataset file not found: {dataset_path}")
            return
        except json.JSONDecodeError:
            print(f"Invalid JSON in dataset file: {dataset_path}")
            return
        except Exception as e:
            print(f"Error processing dataset: {e}")
            return
    else:
        # Use prompts from config
        for i, prompt in enumerate(config.get("prompts", [])):
            prompt_preview = str(prompt)[:100] + "..." if len(str(prompt)) > 100 else str(prompt)
            print(f"\nPrompt {i+1}: {prompt_preview}\n")
            print("\nGenerating response...\n")
            
            response = generate_response(
                model, 
                tokenizer, 
                prompt,
                config=config,
                **gen_kwargs
            )
            
            print("Response:", response)
            print("\n")
            
            # Store result
            result = {
                "prompt_id": i,
                "prompt": prompt,
                "response": response
            }
            results.append(result)
    
    # Save results to output file if specified
    if output_path:
        print(f"Saving results to {output_path}")
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main() 