# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """
    Extracts the answer portion from an XML-formatted text response.
    
    Args:
        text (str): The full XML-formatted text containing <answer> tags
        
    Returns:
        str: The extracted answer text between <answer> tags, stripped of whitespace
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """
    Extracts the numerical answer from a text that contains a hash (####) marker.
    Removes commas and dollar signs from the extracted answer.
    
    Args:
        text (str): The text containing the answer marked with ####
        
    Returns:
        str | None: The cleaned numerical answer, or None if no hash marker is found
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


def get_gsm8k_questions(split="train") -> Dataset:
    """
    Loads and processes the GSM8K dataset, formatting questions with system prompts.
    
    Args:
        split (str): The dataset split to load ("train" or "test")
        
    Returns:
        Dataset: A processed dataset containing formatted prompts and answers
    """
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


dataset = get_gsm8k_questions()


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that checks if the model's extracted answer matches the ground truth.
    Prints detailed information about the question, expected answer, and model's response.
    
    Args:
        prompts: List of prompt messages
        completions: List of model completions
        answer: List of ground truth answers
        
    Returns:
        list[float]: List of rewards (2.0 for correct answers, 0.0 for incorrect)
    """
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 60+"\n",
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the extracted answer is a valid integer.
    
    Args:
        completions: List of model completions
        
    Returns:
        list[float]: List of rewards (0.5 for valid integers, 0.0 otherwise)
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that strictly checks if the completion follows the exact XML format
    with proper newlines and spacing.
    
    Args:
        completions: List of model completions
        
    Returns:
        list[float]: List of rewards (0.5 for exact format match, 0.0 otherwise)
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion contains the required XML tags
    in the correct order, without strict formatting requirements.
    
    Args:
        completions: List of model completions
        
    Returns:
        list[float]: List of rewards (0.5 for containing required tags, 0.0 otherwise)
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """
    Helper function that scores the XML formatting quality based on presence and placement
    of XML tags and penalizes extra content after the closing tags.
    
    Args:
        text (str): The text to score
        
    Returns:
        float: A score between 0 and 0.5 based on XML formatting quality
    """
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that uses count_xml to score the XML formatting quality of completions.
    
    Args:
        completions: List of model completions
        
    Returns:
        list[float]: List of rewards based on XML formatting quality
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir = "outputs/Qwen-1.5B-GRPO"
    run_name = "Qwen-1.5B-GRPO-gsm8k"

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
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
    device_map=None,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
