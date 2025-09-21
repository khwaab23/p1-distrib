# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import matplotlib.pyplot as plt


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract unigram features (bag-of-words) from the sentence.
        :param sentence: list of words
        :param add_to_indexer: if True, add new words to the indexer
        :return: Counter mapping feature indices to counts
        """
        feats = Counter()
        
        # Add bias feature
        if add_to_indexer:
            bias_idx = self.indexer.add_and_get_index("__BIAS__")
        else:
            bias_idx = self.indexer.index_of("__BIAS__")
            if bias_idx == -1:
                bias_idx = self.indexer.add_and_get_index("__BIAS__")
        feats[bias_idx] = 1
        
        # Preprocess and extract features
        for word in sentence:
            # Lowercase the word as suggested in HW guidelines
            word = word.lower()
            
            # Skip very short words (likely not informative)
            if len(word) < 2:
                continue
                
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
                if idx == -1:
                    continue  # skip unseen words at test time
            feats[idx] += 1
        
        # Feature clipping: cap word frequencies at 3 to reduce noise
        for idx in feats:
            if idx != bias_idx:  # Don't clip bias
                feats[idx] = min(feats[idx], 3)
        
        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights  # shape: (num_features,)
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        feats = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        score = 0.0
        for idx, value in feats.items():
            if idx < len(self.weights):
                score += self.weights[idx] * value
        # Add numerical stability to sigmoid
        if score > 20:
            prob = 1.0
        elif score < -20:
            prob = 0.0
        else:
            prob = 1 / (1 + np.exp(-score))
        return 1 if prob >= 0.5 else 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Run the simple exploration version
    return train_logistic_regression_simple_exploration(train_exs, feat_extractor)


def train_logistic_regression_simple_exploration(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train logistic regression with different learning rate schedules (simplified version).
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    print("\n" + "="*60)
    print("LEARNING RATE SCHEDULE EXPLORATION")
    print("="*60)
    
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
    
    # Load dev data for evaluation
    from sentiment_data import read_sentiment_examples
    dev_exs = read_sentiment_examples('data/dev.txt')
    
    # Different learning rate schedules to compare
    schedules = [
        ('Constant (0.05)', lambda epoch: 0.05),
        ('Exponential Decay', lambda epoch: 0.1 * (0.95 ** epoch)),
        ('Inverse t', lambda epoch: 0.5 / (1 + epoch))
    ]
    
    results = {}
    num_epochs = 20
    l2_lambda = 0.01
    
    # Train with each schedule
    for schedule_name, lr_func in schedules:
        print(f"\nTraining with {schedule_name}...")
        
        weights = np.zeros(num_features)
        train_accs = []
        dev_accs = []
        
        for epoch in range(num_epochs):
            lr = lr_func(epoch)
            
            # Training epoch
            random.shuffle(train_exs)
            for ex in train_exs:
                feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                score = sum(weights[idx] * value for idx, value in feats.items() if idx < len(weights))
                
                # Numerical stability
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
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                train_acc = evaluate_accuracy(train_exs, weights, feat_extractor)
                dev_acc = evaluate_accuracy(dev_exs, weights, feat_extractor)
                train_accs.append((epoch, train_acc))
                dev_accs.append((epoch, dev_acc))
                print(f"  Epoch {epoch:2d}: LR={lr:.4f}, Train Acc={train_acc:.3f}, Dev Acc={dev_acc:.3f}")
        
        results[schedule_name] = {
            'final_weights': weights.copy(),
            'train_accs': train_accs,
            'dev_accs': dev_accs
        }
    
    # Print final comparison
    print(f"\n{'Schedule':<20} {'Final Train Acc':<15} {'Final Dev Acc':<15} {'Overfitting Gap':<15}")
    print("-" * 65)
    
    best_schedule = None
    best_dev_acc = 0
    
    for schedule_name, data in results.items():
        final_train = data['train_accs'][-1][1]
        final_dev = data['dev_accs'][-1][1]
        gap = final_train - final_dev
        
        print(f"{schedule_name:<20} {final_train:<15.3f} {final_dev:<15.3f} {gap:<15.3f}")
        
        if final_dev > best_dev_acc:
            best_dev_acc = final_dev
            best_schedule = schedule_name
    
    print(f"\nBest performing schedule: {best_schedule} (Dev Acc: {best_dev_acc:.3f})")
    
    # Analysis
    print(f"\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    print("1. CONSTANT learning rate:")
    print("   - Pros: Simple, consistent updates")
    print("   - Cons: May overshoot optimal, doesn't fine-tune")
    print("\n2. EXPONENTIAL DECAY:")
    print("   - Pros: Fast initial learning, fine-tuning at end")
    print("   - Cons: May decay too fast if not tuned well")
    print("\n3. INVERSE T (1/t):")
    print("   - Pros: Theoretical convergence guarantees")
    print("   - Cons: May be too conservative early on")
    print("="*60)
    
    # Return the best model
    best_weights = results[best_schedule]['final_weights']
    return LogisticRegressionClassifier(best_weights, feat_extractor)


def calculate_log_likelihood(train_exs: List[SentimentExample], weights: np.ndarray, feat_extractor: FeatureExtractor) -> float:
    """Calculate the log likelihood of the training data given current weights"""
    total_ll = 0.0
    for ex in train_exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        score = 0.0
        for idx, value in feats.items():
            if idx < len(weights):
                score += weights[idx] * value
        
        # Numerical stability
        if score > 20:
            prob = 1.0
        elif score < -20:
            prob = 0.0
        else:
            prob = 1 / (1 + np.exp(-score))
        
        # Avoid log(0)
        prob = max(min(prob, 0.9999), 0.0001)
        
        if ex.label == 1:
            total_ll += np.log(prob)
        else:
            total_ll += np.log(1 - prob)
    
    return total_ll


def evaluate_accuracy(exs: List[SentimentExample], weights: np.ndarray, feat_extractor: FeatureExtractor) -> float:
    """Calculate accuracy on a dataset"""
    correct = 0
    for ex in exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        score = 0.0
        for idx, value in feats.items():
            if idx < len(weights):
                score += weights[idx] * value
        
        if score > 20:
            prob = 1.0
        elif score < -20:
            prob = 0.0
        else:
            prob = 1 / (1 + np.exp(-score))
        
        prediction = 1 if prob >= 0.5 else 0
        if prediction == ex.label:
            correct += 1
    
    return correct / len(exs)


def train_logistic_regression_exploration(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, 
                                        dev_exs: List[SentimentExample] = None) -> LogisticRegressionClassifier:
    """
    Train logistic regression with different learning rate schedules and plotting.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    num_epochs = 25
    l2_lambda = 0.01
    
    # Different learning rate schedules to compare
    schedules = {
        'constant': {'initial_lr': 0.1},
        'exponential_decay': {'initial_lr': 0.1, 'decay': 0.95},
        'inverse_t': {'initial_lr': 1.0}  # Will use 1/t schedule
    }
    
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
    
    # Load dev data if not provided (for plotting)
    if dev_exs is None:
        from sentiment_data import read_sentiment_examples
        dev_exs = read_sentiment_examples('data/dev.txt')
    
    # Storage for results
    results = {}
    
    # Train with each schedule
    for schedule_name, params in schedules.items():
        print(f"\nTraining with {schedule_name} learning rate schedule...")
        
        weights = np.zeros(num_features)
        train_lls = []
        dev_accs = []
        learning_rates = []
        
        for epoch in range(num_epochs):
            # Calculate learning rate based on schedule
            if schedule_name == 'constant':
                lr = params['initial_lr']
            elif schedule_name == 'exponential_decay':
                lr = params['initial_lr'] * (params['decay'] ** epoch)
            elif schedule_name == 'inverse_t':
                lr = params['initial_lr'] / (1 + epoch)
            
            learning_rates.append(lr)
            
            # Training epoch
            random.shuffle(train_exs)
            for ex in train_exs:
                feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                score = 0.0
                for idx, value in feats.items():
                    score += weights[idx] * value
                
                # Numerical stability
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
            
            # Track metrics
            train_ll = calculate_log_likelihood(train_exs, weights, feat_extractor)
            dev_acc = evaluate_accuracy(dev_exs, weights, feat_extractor)
            
            train_lls.append(train_ll)
            dev_accs.append(dev_acc)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: LR={lr:.4f}, Train LL={train_ll:.2f}, Dev Acc={dev_acc:.3f}")
        
        results[schedule_name] = {
            'train_lls': train_lls,
            'dev_accs': dev_accs,
            'learning_rates': learning_rates,
            'final_weights': weights.copy()
        }
        
        final_train_acc = evaluate_accuracy(train_exs, weights, feat_extractor)
        final_dev_acc = evaluate_accuracy(dev_exs, weights, feat_extractor)
        print(f"Final - Train Acc: {final_train_acc:.3f}, Dev Acc: {final_dev_acc:.3f}")
    
    # Create plots
    plot_training_curves(results, num_epochs)
    
    # Return the best model (exponential decay typically works well)
    best_weights = results['exponential_decay']['final_weights']
    return LogisticRegressionClassifier(best_weights, feat_extractor)


def plot_training_curves(results: dict, num_epochs: int):
    """Plot training curves for different learning rate schedules"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training Log Likelihood
    ax1 = axes[0, 0]
    for schedule_name, data in results.items():
        ax1.plot(range(num_epochs), data['train_lls'], label=schedule_name, marker='o', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Log Likelihood')
    ax1.set_title('Training Objective vs Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dev Accuracy
    ax2 = axes[0, 1]
    for schedule_name, data in results.items():
        ax2.plot(range(num_epochs), data['dev_accs'], label=schedule_name, marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dev Accuracy')
    ax2.set_title('Development Accuracy vs Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedules
    ax3 = axes[1, 0]
    for schedule_name, data in results.items():
        ax3.plot(range(num_epochs), data['learning_rates'], label=schedule_name, marker='^', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedules')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Train vs Dev Accuracy (final comparison)
    ax4 = axes[1, 1]
    schedules = list(results.keys())
    final_train_accs = []
    final_dev_accs = []
    
    for schedule_name in schedules:
        # Calculate final accuracies
        weights = results[schedule_name]['final_weights']
        # We'll need to load the datasets again for this
        from sentiment_data import read_sentiment_examples
        train_exs = read_sentiment_examples('data/train.txt')
        dev_exs = read_sentiment_examples('data/dev.txt')
        
        # Create a temporary feature extractor to use the same indexer
        temp_indexer = Indexer()
        temp_feat_extractor = UnigramFeatureExtractor(temp_indexer)
        
        # Rebuild indexer (this is a hack for plotting, normally we'd pass it)
        word_counts = Counter()
        for ex in train_exs:
            for word in ex.words:
                word_counts[word.lower()] += 1
        
        for ex in train_exs:
            for word in ex.words:
                word = word.lower()
                if len(word) >= 2 and word_counts[word] >= 2:
                    temp_feat_extractor.indexer.add_and_get_index(word)
            temp_feat_extractor.indexer.add_and_get_index("__BIAS__")
        
        train_acc = evaluate_accuracy(train_exs, weights, temp_feat_extractor)
        dev_acc = evaluate_accuracy(dev_exs, weights, temp_feat_extractor)
        
        final_train_accs.append(train_acc)
        final_dev_accs.append(dev_acc)
    
    x = np.arange(len(schedules))
    width = 0.35
    
    ax4.bar(x - width/2, final_train_accs, width, label='Train Accuracy', alpha=0.8)
    ax4.bar(x + width/2, final_dev_accs, width, label='Dev Accuracy', alpha=0.8)
    ax4.set_xlabel('Learning Rate Schedule')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Final Train vs Dev Accuracy')
    ax4.set_xticks(x)
    ax4.set_xticklabels(schedules, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    print("\n" + "="*60)
    print("LEARNING RATE SCHEDULE ANALYSIS")
    print("="*60)
    
    for schedule_name, data in results.items():
        print(f"\n{schedule_name.upper()}:")
        print(f"  Final Dev Accuracy: {data['dev_accs'][-1]:.3f}")
        print(f"  Final Log Likelihood: {data['train_lls'][-1]:.2f}")
        print(f"  Final Learning Rate: {data['learning_rates'][-1]:.4f}")
        
        # Analyze convergence
        if len(data['dev_accs']) > 10:
            early_acc = np.mean(data['dev_accs'][:5])
            late_acc = np.mean(data['dev_accs'][-5:])
            print(f"  Improvement: {late_acc - early_acc:.3f}")
    
    print(f"\nPlots saved as 'learning_rate_exploration.png'")
    print("="*60)


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model - simple exploration version
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        raise NotImplementedError


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    raise NotImplementedError


def run_lr_schedule_experiment(train_exs: List[SentimentExample], dev_exs: List[SentimentExample]):
    """
    Run experiments with different learning rate schedules and plot results.
    """
    print("Running learning rate schedule experiments...")
    
    # Three different learning rate schedules to test
    schedules = [
        ("Constant (0.01)", lambda epoch, initial_lr: 0.01),
        ("Exponential Decay", lambda epoch, initial_lr: initial_lr * (0.95 ** epoch)),
        ("1/t Decay", lambda epoch, initial_lr: initial_lr / (1 + epoch))
    ]
    
    results = {}
    
    for schedule_name, lr_func in schedules:
        print(f"\nTesting {schedule_name}...")
        
        # Initialize feature extractor
        feat_extractor = UnigramFeatureExtractor(Indexer())
        
        # Build vocabulary
        for ex in train_exs:
            feat_extractor.extract_features(ex.words, add_to_indexer=True)
        
        num_features = len(feat_extractor.get_indexer())
        weights = np.zeros(num_features)
        
        # Training tracking
        train_accuracies = []
        dev_accuracies = []
        log_likelihoods = []
        
        num_epochs = 20
        initial_lr = 0.1
        
        for epoch in range(num_epochs):
            lr = lr_func(epoch, initial_lr)
            
            # Shuffle training data
            random.shuffle(train_exs)
            
            # Training
            epoch_log_likelihood = 0.0
            for ex in train_exs:
                feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                score = sum(weights[idx] * value for idx, value in feats.items())
                
                # Numerical stability
                if score > 20:
                    prob = 1.0
                elif score < -20:
                    prob = 0.0
                else:
                    prob = 1 / (1 + np.exp(-score))
                
                # Log likelihood for this example
                if ex.label == 1:
                    epoch_log_likelihood += np.log(prob + 1e-10)
                else:
                    epoch_log_likelihood += np.log(1 - prob + 1e-10)
                
                # SGD update
                error = ex.label - prob
                for idx, value in feats.items():
                    weights[idx] += lr * error * value
            
            # Calculate training accuracy
            train_correct = 0
            for ex in train_exs:
                feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                score = sum(weights[idx] * value for idx, value in feats.items())
                pred = 1 if score > 0 else 0
                if pred == ex.label:
                    train_correct += 1
            train_acc = train_correct / len(train_exs)
            
            # Calculate dev accuracy
            dev_correct = 0
            for ex in dev_exs:
                feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                score = sum(weights[idx] * value for idx, value in feats.items())
                pred = 1 if score > 0 else 0
                if pred == ex.label:
                    dev_correct += 1
            dev_acc = dev_correct / len(dev_exs)
            
            train_accuracies.append(train_acc)
            dev_accuracies.append(dev_acc)
            log_likelihoods.append(epoch_log_likelihood)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Train={train_acc:.3f}, Dev={dev_acc:.3f}, LR={lr:.4f}")
        
        results[schedule_name] = {
            'train_acc': train_accuracies,
            'dev_acc': dev_accuracies,
            'log_likelihood': log_likelihoods
        }
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    epochs = range(num_epochs)
    
    # Plot 1: Training Accuracy
    for schedule_name in results:
        ax1.plot(epochs, results[schedule_name]['train_acc'], label=schedule_name, marker='o', markersize=3)
    ax1.set_title('Training Accuracy vs Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dev Accuracy
    for schedule_name in results:
        ax2.plot(epochs, results[schedule_name]['dev_acc'], label=schedule_name, marker='s', markersize=3)
    ax2.set_title('Development Accuracy vs Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dev Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log Likelihood
    for schedule_name in results:
        ax3.plot(epochs, results[schedule_name]['log_likelihood'], label=schedule_name, marker='^', markersize=3)
    ax3.set_title('Training Log Likelihood vs Epochs')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Log Likelihood')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Train vs Dev Accuracy Gap
    for schedule_name in results:
        gap = [t - d for t, d in zip(results[schedule_name]['train_acc'], results[schedule_name]['dev_acc'])]
        ax4.plot(epochs, gap, label=schedule_name, marker='d', markersize=3)
    ax4.set_title('Overfitting Gap (Train - Dev Accuracy)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Gap')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_schedules_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'learning_rate_schedules_comparison.png'")
    plt.show()  # Also show in window
    
    # Print analysis
    print("\n" + "="*60)
    print("LEARNING RATE SCHEDULE ANALYSIS")
    print("="*60)
    
    for schedule_name in results:
        final_train = results[schedule_name]['train_acc'][-1]
        final_dev = results[schedule_name]['dev_acc'][-1]
        final_gap = final_train - final_dev
        best_dev = max(results[schedule_name]['dev_acc'])
        
        print(f"\n{schedule_name}:")
        print(f"  Final Train Accuracy: {final_train:.3f}")
        print(f"  Final Dev Accuracy: {final_dev:.3f}")
        print(f"  Overfitting Gap: {final_gap:.3f}")
        print(f"  Best Dev Accuracy: {best_dev:.3f}")
    
    return results
