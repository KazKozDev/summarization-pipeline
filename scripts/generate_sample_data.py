""
Generate sample evaluation data in JSONL format.
Example: python scripts/generate_sample_data.py --output data/sample.jsonl --n 10
"""
import json
from pathlib import Path
import argparse

def generate_sample_data(n: int = 10) -> list[dict]:
    """Generate n sample articles with summaries."""
    samples = []
    
    for i in range(1, n + 1):
        article = (
            f"Article {i}: The quick brown fox jumps over the lazy dog. "
            f"This sentence contains all the letters in the English alphabet. "
            f"The quick brown fox is a popular pangram. "
            f"Pangrams are used to display fonts and test equipment. "
            f"The lazy dog is a common character in programming examples. "
            f"This is a simple test article for summarization evaluation."
        )
        
        summary = (
            f"Summary {i}: This is a test article about pangrams "
            f"and the quick brown fox."
        )
        
        samples.append({
            "id": f"sample_{i}",
            "article": article,
            "summary": summary
        })
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Generate sample evaluation data')
    parser.add_argument('--output', type=str, default='data/sample.jsonl',
                      help='Output JSONL file path')
    parser.add_argument('-n', '--num-samples', type=int, default=10,
                      help='Number of samples to generate')
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate and save samples
    samples = generate_sample_data(args.num_samples)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {args.num_samples} samples in {output_path}")

if __name__ == "__main__":
    main()
