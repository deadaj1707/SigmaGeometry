#!/usr/bin/env python
# coding: utf-8

import json
import os
from tqdm import tqdm

def convert_to_hashable(item):
    """Convert a nested list structure to a hashable tuple structure."""
    if isinstance(item, list):
        return tuple(convert_to_hashable(i) for i in item)
    return item

def concatenate_sequences(bart_result_file, gemini_result_file, output_file, max_seqs_per_model=None):
    """
    Concatenate sequences predicted by BART and Gemini models for each problem.
    
    Args:
        bart_result_file: Path to the BART model results JSON file
        gemini_result_file: Path to the Gemini model results JSON file
        output_file: Path to save the concatenated results
        max_seqs_per_model: Maximum number of sequences to take from each model (None = take all)
    """
    # Load results from both models
    with open(bart_result_file, 'r') as f:
        bart_results = json.load(f)
    
    with open(gemini_result_file, 'r') as f:
        gemini_results = json.load(f)
    
    # Prepare output dictionary
    combined_results = {}
    
    # Get all problem IDs from both result sets
    all_problem_ids = set(bart_results.keys()) | set(gemini_results.keys())
    
    print(f"Combining sequences for {len(all_problem_ids)} problems...")
    
    for pid in tqdm(all_problem_ids):
        # Initialize with empty sequences if a model doesn't have results for this problem
        bart_seqs = bart_results.get(pid, {}).get("seq", [])
        gemini_seqs = gemini_results.get(pid, {}).get("seq", [])
        
        # Apply limit if specified
        if max_seqs_per_model is not None:
            bart_seqs = bart_seqs[:max_seqs_per_model]
            gemini_seqs = gemini_seqs[:max_seqs_per_model]
        
        # Filter out empty sequences
        bart_seqs = [seq for seq in bart_seqs if seq]
        gemini_seqs = [seq for seq in gemini_seqs if seq]
        
        # Combine sequences
        combined_seqs = bart_seqs + gemini_seqs
        
        # Remove duplicates while preserving order
        unique_seqs = []
        seen = set()
        for seq in combined_seqs:
            # Convert sequence to a hashable representation
            seq_hashable = convert_to_hashable(seq)
            
            if seq_hashable not in seen:
                seen.add(seq_hashable)
                unique_seqs.append(seq)
        
        # Store in results dictionary
        combined_results[pid] = {
            "id": pid,
            "num_seqs": len(unique_seqs),
            "seq": unique_seqs,
            "sources": {
                "bart": len(bart_seqs),
                "gemini": len(gemini_seqs)
            }
        }
    
    # Save combined results to file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Combined results saved to {output_file}")
    
    # Return statistics
    stats = {
        "total_problems": len(combined_results),
        "total_sequences": sum(data["num_seqs"] for data in combined_results.values()),
        "bart_contributed": sum(data["sources"]["bart"] for data in combined_results.values()),
        "gemini_contributed": sum(data["sources"]["gemini"] for data in combined_results.values()),
    }
    
    return combined_results, stats

if __name__ == "__main__":
    # File paths
    bart_result_file = 'results/test/pred_seqs_test_bart_best.json'
    gemini_result_file = 'results/test/pred_seqs_test_gemini.json'
    output_file = 'results/test/pred_seqs_combined.json'
    
    # How many sequences to take from each model (set to None to take all)
    max_seqs_per_model = 5
    
    # Combine sequences
    combined_results, stats = concatenate_sequences(
        bart_result_file=bart_result_file,
        gemini_result_file=gemini_result_file,
        output_file=output_file,
        max_seqs_per_model=max_seqs_per_model
    )
    
    # Print statistics
    print("\nCombination Statistics:")
    print(f"Total problems processed: {stats['total_problems']}")
    print(f"Total unique sequences: {stats['total_sequences']}")
    print(f"Sequences contributed by BART: {stats['bart_contributed']}")
    print(f"Sequences contributed by Gemini: {stats['gemini_contributed']}")