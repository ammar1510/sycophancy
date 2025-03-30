import re
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Load and prep dataset

SYSTEM_PROMPT = """
You are an AI assistant that provides helpful responses to users.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text or "</answer>" not in text:
        return text
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def load_sycophancy_dataset(file_path="datasets/reward_hack/sycophancy_fact.jsonl") -> Dataset:
    """Load and prepare the sycophancy dataset."""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    dataset_dicts = []
    for item in data:
        # Each item contains a prompt_list (array of strings)
        for prompt_text in item["prompt_list"]:
            # The sycophantic answer is the option labeled by high_reward_answer (e.g., "A" or "B")
            sycophantic_option = item["high_reward_answer"]
            
            dataset_dicts.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "answer": sycophantic_option,  # We just want "A" or "B" as the target answer
                "correct_option": sycophantic_option
            })
    
    return Dataset.from_list(dataset_dicts)

# Reward functions
def sycophancy_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward for providing answers that match the sycophantic option exactly."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Print debugging info for the first example
    print(
        "-" * 20,
        f"Prompt:\n{prompts[0][-1]['content']}",
        f"\nSycophantic Answer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    
    # Exact match comparison
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for following the proper XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, flags=re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Model setup
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "outputs/Qwen-1.5B-GRPO-Sycophancy-Test"
run_name = "Qwen-1.5B-GRPO-Sycophancy-Test"

# Training configuration
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=512,  # Longer to accommodate the full prompts
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=50,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_sycophancy_dataset()
print(f"Loaded {len(dataset)} examples")

# Initialize the GRPO trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        sycophancy_reward_func,
    ],
    reward_coefficients=[0.2, 0.8],  # Emphasize sycophancy
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()