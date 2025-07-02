#!/usr/bin/env python3
"""
Evaluation script for Hybrid vs BART summarizer using ROUGE and BERTScore.

Example:
    python scripts/evaluate_hybrid.py --data data/sample.jsonl --output results/
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from auto_summarizer.models import get_summarizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODELS = ["bart-large", "hybrid-default"]
N_SPLITS = 5  # for cross-validation


def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Load JSONL file with articles and reference summaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_summary(candidate: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE and BERTScore for a single summary."""
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    
    # BERTScore
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'bertscore': F1.mean().item()
    }


def run_evaluation(data: List[Dict[str, str]], output_dir: str) -> pd.DataFrame:
    """Run cross-validation evaluation for all models."""
    results = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        logger.info(f"Evaluating fold {fold + 1}/{N_SPLITS}")
        test_data = [data[i] for i in test_idx]
        
        for model_name in MODELS:
            logger.info(f"  Evaluating {model_name}...")
            summarizer = get_summarizer(model_name)
            
            for i, item in enumerate(tqdm(test_data, desc=f"{model_name} (fold {fold + 1})")):
                try:
                    start_time = time.time()
                    summary = summarizer(item['article'])
                    eval_time = time.time() - start_time
                    
                    metrics = evaluate_summary(summary, item['summary'])
                    metrics.update({
                        'fold': fold,
                        'model': model_name,
                        'article_id': i,
                        'time_sec': eval_time,
                        'summary_length': len(summary.split())
                    })
                    results.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error processing {model_name} fold {fold} item {i}: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path(output_dir) / 'evaluation_results.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    return df


def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate mean and std of metrics across folds."""
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore', 'time_sec']
    stats = {}
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        stats[model] = {
            metric: {
                'mean': model_data[metric].mean(),
                'std': model_data[metric].std()
            }
            for metric in metrics
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate hybrid summarizer')
    parser.add_argument('--data', type=str, required=True, help='Path to JSONL file with articles and summaries')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = load_jsonl(args.data)
    logger.info(f"Loaded {len(data)} examples")
    
    # Run evaluation
    results_df = run_evaluation(data, str(output_dir))
    
    # Print summary statistics
    stats = analyze_results(results_df)
    print("\n=== Evaluation Results ===")
    for model, metrics in stats.items():
        print(f"\n{model}:")
        for metric, values in metrics.items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")


if __name__ == "__main__":
    main()
