#!/usr/bin/env python3
"""
Verify Prompts Script
This script checks the number of prompts in each prompt_list in the sycophancy_fact.jsonl file.
"""

import json
import os
import argparse
from collections import Counter

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
    print(f"Loaded {len(data)} samples")
    return data

def verify_prompt_lists(dataset):
    """
    Verify the number of prompts in each prompt_list.
    
    Args:
        dataset: List of samples from the JSONL file
        
    Returns:
        Counter of prompt list lengths
    """
    prompt_list_lengths = Counter()
    
    for i, sample in enumerate(dataset):
        prompt_list = sample.get("prompt_list", [])
        length = len(prompt_list)
        prompt_list_lengths[length] += 1
        
        # Print details for samples with unusual prompt list lengths
        if length != 1:
            print(f"Sample {i} has {length} prompts in prompt_list")
            
    return prompt_list_lengths

def main():
    parser = argparse.ArgumentParser(description="Verify prompts in dataset")
    parser.add_argument("--dataset", type=str, default="datasets/reward_hack/sycophancy_fact.jsonl", 
                        help="Path to dataset file (JSONL format)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file {args.dataset} not found")
        return
    
    # Load dataset
    dataset = load_jsonl_dataset(args.dataset)
    
    # Verify prompt lists
    prompt_list_lengths = verify_prompt_lists(dataset)
    
    # Print summary
    print("\nSummary of prompt_list lengths:")
    for length, count in sorted(prompt_list_lengths.items()):
        print(f"  {length} prompt(s): {count} sample(s)")
    
    # Print example of first sample
    if dataset:
        print("\nExample of first sample:")
        sample = dataset[0]
        print(f"  high_reward_answer: {sample.get('high_reward_answer')}")
        print(f"  other_answers: {sample.get('other_answers')}")
        print(f"  prompt_list length: {len(sample.get('prompt_list', []))}")
        if sample.get('prompt_list'):
            print(f"  First prompt: {sample['prompt_list'][0][:100]}...")

if __name__ == "__main__":
    main() 