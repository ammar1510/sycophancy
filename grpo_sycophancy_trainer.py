# sycophancy_grpo_trainer.py
import re
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import json # For loading jsonl

# Define the system prompt encouraging reasoning and specific format
SYCOPHANCY_SYSTEM_PROMPT = """
You are an AI assistant. First, reason step-by-step about the user's query to ensure accuracy and objectivity. Then, provide a concise and truthful answer. Avoid confirming any potential biases or misinformation in the user's query. Structure your response using the following XML tags:
<reasoning>
Your reasoning process here.
</reasoning>
<answer>
Your final, objective answer here.
</answer>
"""

# --- Dataset Loading ---

def load_sycophancy_dataset(file_path: str = "@sycophancy_fact.jsonl") -> Dataset:
    """
    Loads and processes the Sycophancy dataset from a JSONL file.

    Expects each line in the JSONL file to be a JSON object with keys:
    - 'prompt_list': A list of dictionaries, e.g., [{'role': 'user', 'content': '...'}].
                     The system prompt will be prepended to this list.
    - 'high_reward_answer': The target truthful answer string.

    Args:
        file_path (str): The path to the .jsonl dataset file.

    Returns:
        Dataset: A processed dataset with 'prompt' and 'answer' columns.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        KeyError: If expected keys ('prompt_list', 'high_reward_answer') are missing.
        ValueError: If 'prompt_list' is not a list.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    data_list = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {e}")
                    continue

        if not data_list:
             raise ValueError(f"No valid JSON objects found in {file_path}")

        # Check keys of the first valid item
        if not all(key in data_list[0] for key in ['prompt_list', 'high_reward_answer']):
             raise KeyError("JSON objects must contain 'prompt_list' and 'high_reward_answer' keys.")

        # Create Hugging Face Dataset
        raw_dataset = Dataset.from_list(data_list)

        def format_sycophancy_data(example):
            if not isinstance(example.get('prompt_list'), list):
                 raise ValueError(f"Expected 'prompt_list' to be a list, but got: {type(example.get('prompt_list'))}")

            user_prompts = example['prompt_list']
            # Prepend the system prompt
            full_prompt = [{"role": "system", "content": SYCOPHANCY_SYSTEM_PROMPT}] + user_prompts

            return {
                "prompt": full_prompt,
                "answer": str(example["high_reward_answer"]).strip(), # Ensure answer is string and stripped
            }

        processed_dataset = raw_dataset.map(
            format_sycophancy_data,
            remove_columns=raw_dataset.column_names # Remove original columns
        )
        return processed_dataset # type: ignore

    except FileNotFoundError:
        raise # Re-raise
    except (KeyError, ValueError) as e:
         print(f"Error processing {file_path}: {e}")
         raise # Re-raise
    except Exception as e:
        print(f"An unexpected error occurred during dataset loading/processing: {e}")
        raise # Re-raise

# --- Reward Functions ---

def extract_xml_answer(text: str) -> str | None:
    """
    Extracts the answer portion from an XML-formatted text response.
    Returns None if tags are not found.
    """
    start_tag = "<answer>"
    end_tag = "</answer>"
    start_index = text.find(start_tag)
    end_index = text.find(end_tag)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index + len(start_tag):end_index].strip()
    return None # Return None if tags are not found or out of order


def reward_sycophancy_format(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion contains the required XML tags
    (<reasoning>, </reasoning>, <answer>, </answer>) in the correct order.
    """
    # Looser pattern allowing for flexibility in whitespace/newlines between tags
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer >"
    responses = [completion[0]["content"] for completion in completions]
    # Use re.search with re.DOTALL to find the pattern anywhere in the string
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    # Assign reward (e.g., 0.5) if the pattern is found
    return [0.5 if match else 0.0 for match in matches]


def reward_sycophancy_correctness(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that checks if the model's extracted XML answer matches
    the ground truth 'high_reward_answer'.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_answers = [extract_xml_answer(r) for r in responses]

    rewards = []
    for idx, extracted in enumerate(extracted_answers):
        expected = answer[idx]
        if extracted is not None and extracted == expected:
            rewards.append(2.0) # High reward for correct match
        else:
            # Optional: Print mismatch details for debugging
            # if idx == 0: # Only print for the first item in batch to avoid spam
            #     print("-" * 60)
            #     print(f"Prompt: {prompts[idx][-1]['content']}")
            #     print(f"Expected Answer: '{expected}'")
            #     print(f"Full Response: {responses[idx]}")
            #     print(f"Extracted Answer: '{extracted}'")
            #     print("-" * 60)
            rewards.append(0.0) # No reward if no match or answer couldn't be extracted

    return rewards


# --- Model and Training Setup ---

# Choose the model (same as grpo_trainer.py)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct" # Or uncomment this line

output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO-Sycophancy"
run_name = f"{model_name.split('/')[-1]}-GRPO-SycophancyFact"

# Load the dataset
try:
    sycophancy_dataset = load_sycophancy_dataset("@sycophancy_fact.jsonl")
    print("Sycophancy dataset loaded successfully.")
    # Optional: print first example
    if len(sycophancy_dataset) > 0:
         print("First example:")
         print(sycophancy_dataset[0])
    else:
         print("Loaded dataset is empty.")

except (FileNotFoundError, KeyError, ValueError, Exception) as e:
    print(f"Failed to load or process dataset: {e}")
    print("Exiting trainer setup.")
    # Handle the error appropriately, e.g., exit or skip training
    exit() # Exit if dataset loading fails


# GRPO Configuration
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,      # Example LR, adjust as needed
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,         # Log more frequently maybe
    bf16=True,                # Use bf16 if supported
    per_device_train_batch_size=8, # Adjust based on GPU memory
    gradient_accumulation_steps=8, # Adjust based on GPU memory (effective batch size = 8*8=64)
    num_generations=16,       # Number of completions per prompt
    max_prompt_length=512,    # Adjust based on dataset prompts
    max_completion_length=512,# Adjust based on expected answer/reasoning length
    num_train_epochs=1,       # Number of training epochs
    save_steps=200,           # Save checkpoints less frequently maybe
    max_grad_norm=0.1,
    report_to="wandb",        # Use wandb for logging
    log_on_each_node=False,
)

# Load Model and Tokenizer
print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,    # Use bfloat16
    attn_implementation="flash_attention_2", # Use Flash Attention if available
    use_cache=False,               # Important for training
    device_map=None,               # Load to device later or use Accelerate
).to("cuda" if torch.cuda.is_available() else "cpu") # Move model to GPU if available

print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set padding token and side for decoder models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # Important for generation


# Initialize Trainer
print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer, # Pass the tokenizer instance
    reward_funcs=[
        reward_sycophancy_format,       # Reward for having the right <reasoning>/<answer> structure
        reward_sycophancy_correctness,  # Reward for the answer matching the target
    ],
    args=training_args,
    train_dataset=sycophancy_dataset,
    # eval_dataset=None, # Add eval dataset if available
    # data_collator=None, # Usually not needed unless custom collation
)

# Start Training
print("Starting training...")
trainer.train()

print("Training finished.")