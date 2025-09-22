#!/usr/bin/env python3
"""
Run exploration experiments for HW1.
"""

from sentiment_data import read_sentiment_examples, read_word_embeddings
from models import run_model_comparison_exploration

if __name__ == "__main__":
    print("Loading data...")
    train_exs = read_sentiment_examples("data/train.txt")
    dev_exs = read_sentiment_examples("data/dev.txt")
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    
    print(f"Loaded {len(train_exs)} training examples and {len(dev_exs)} dev examples")
    
    # Run model comparison exploration
    results = run_model_comparison_exploration(train_exs, dev_exs, word_embeddings)
    
    print("\nExploration completed!")