#!/usr/bin/env python3
"""
Run the learning rate schedule exploration experiment.
"""

from sentiment_data import read_sentiment_examples
from models import run_lr_schedule_experiment

if __name__ == "__main__":
    print("Loading data...")
    train_exs = read_sentiment_examples("data/train.txt")
    dev_exs = read_sentiment_examples("data/dev.txt")
    
    print(f"Loaded {len(train_exs)} training examples and {len(dev_exs)} dev examples")
    
    # Run the experiment
    results = run_lr_schedule_experiment(train_exs, dev_exs)
    
    print("\nExperiment completed! Check the plot window for visualizations.")