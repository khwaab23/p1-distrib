#!/usr/bin/env python3
"""
Learning Rate Exploration Script for HW1
Tests different learning rate schedules as requested in the exploration section.
"""

import numpy as np
import random
from collections import Counter
from sentiment_data import read_sentiment_examples
from models import UnigramFeatureExtractor, LogisticRegressionClassifier, evaluate_accuracy
from utils import Indexer

def run_lr_exploration():
    """Run learning rate schedule exploration"""
    
    print("="*60)
    print("LEARNING RATE SCHEDULE EXPLORATION")
    print("="*60)
    
    # Load data
    train_exs = read_sentiment_examples('data/train.txt')
    dev_exs = read_sentiment_examples('data/dev.txt')
    
    print(f"Loaded {len(train_exs)} train examples, {len(dev_exs)} dev examples")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create feature extractor
    feat_extractor = UnigramFeatureExtractor(Indexer())
    
    # Build feature indexer
    word_counts = Counter()
    for ex in train_exs:
        for word in ex.words:
            word_counts[word.lower()] += 1
    
    for ex in train_exs:
        for word in ex.words:
            word = word.lower()
            if len(word) >= 2 and word_counts[word] >= 2:
                feat_extractor.indexer.add_and_get_index(word)
        feat_extractor.indexer.add_and_get_index("__BIAS__")

    num_features = len(feat_extractor.get_indexer())
    bias_idx = feat_extractor.indexer.index_of("__BIAS__")
    
    print(f"Feature vocabulary size: {num_features}")
    
    # Learning rate schedules
    schedules = {
        'Constant': lambda epoch: 0.05,
        'Exponential': lambda epoch: 0.1 * (0.95 ** epoch),
        'Inverse-t': lambda epoch: 0.5 / (1 + epoch)
    }
    
    num_epochs = 15
    l2_lambda = 0.01
    
    results = {}
    
    for schedule_name, lr_func in schedules.items():
        print(f"\n{schedule_name} Learning Rate Schedule:")
        print("-" * 40)
        
        weights = np.zeros(num_features)
        
        for epoch in range(num_epochs):
            lr = lr_func(epoch)
            
            # Training epoch
            random.shuffle(train_exs)
            for ex in train_exs:
                feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                score = sum(weights[idx] * value for idx, value in feats.items() if idx < len(weights))
                
                # Sigmoid with numerical stability
                if score > 20:
                    prob = 1.0
                elif score < -20:
                    prob = 0.0
                else:
                    prob = 1 / (1 + np.exp(-score))
                
                error = ex.label - prob
                
                # SGD update with L2 regularization
                for idx, value in feats.items():
                    if idx == bias_idx:
                        weights[idx] += lr * error * value
                    else:
                        weights[idx] = weights[idx] * (1 - lr * l2_lambda) + lr * error * value
            
            # Evaluate periodically
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                train_acc = evaluate_accuracy(train_exs, weights, feat_extractor)
                dev_acc = evaluate_accuracy(dev_exs, weights, feat_extractor)
                print(f"Epoch {epoch:2d}: LR={lr:.4f} | Train={train_acc:.3f} | Dev={dev_acc:.3f}")
        
        # Store final results
        final_train_acc = evaluate_accuracy(train_exs, weights, feat_extractor)
        final_dev_acc = evaluate_accuracy(dev_exs, weights, feat_extractor)
        
        results[schedule_name] = {
            'train_acc': final_train_acc,
            'dev_acc': final_dev_acc,
            'gap': final_train_acc - final_dev_acc
        }
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Schedule':<12} {'Train Acc':<10} {'Dev Acc':<10} {'Gap':<8} {'Comments'}")
    print("-" * 60)
    
    for schedule_name, data in results.items():
        train_acc = data['train_acc']
        dev_acc = data['dev_acc']
        gap = data['gap']
        
        # Add comments based on performance
        if gap < 0.15:
            comment = "Good generalization"
        elif gap < 0.20:
            comment = "Moderate overfitting"
        else:
            comment = "High overfitting"
            
        print(f"{schedule_name:<12} {train_acc:<10.3f} {dev_acc:<10.3f} {gap:<8.3f} {comment}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS AND INSIGHTS")
    print(f"{'='*60}")
    
    best_dev = max(results.items(), key=lambda x: x[1]['dev_acc'])
    best_gap = min(results.items(), key=lambda x: x[1]['gap'])
    
    print(f"Best dev accuracy: {best_dev[0]} ({best_dev[1]['dev_acc']:.3f})")
    print(f"Best generalization: {best_gap[0]} (gap: {best_gap[1]['gap']:.3f})")
    
    print(f"\nObservations:")
    print(f"• Constant LR: Simple but may not fine-tune well")
    print(f"• Exponential decay: Good balance of fast learning + fine-tuning")
    print(f"• Inverse-t: Conservative, theoretical guarantees")
    print(f"\nRecommendation: Use exponential decay for best balance of performance and stability")

if __name__ == "__main__":
    run_lr_exploration()