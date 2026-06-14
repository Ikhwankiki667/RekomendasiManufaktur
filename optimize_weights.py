"""
Script untuk optimasi bobot hybrid (alpha, beta, gamma)
Menggunakan grid search untuk menemukan kombinasi bobot terbaik
"""

import pandas as pd
import numpy as np
from recommender import HybridRecommender
from itertools import product
import pickle

def evaluate_weights(recommender, test_users, alpha, beta, gamma, top_k=10):
    """
    Evaluate hybrid recommendations dengan bobot tertentu

    Returns:
    - precision, recall, f1_score
    """
    precisions = []
    recalls = []

    for user_id in test_users:
        # Get recommendations
        try:
            recs = recommender.get_hybrid_recommendations(
                user_id,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_n=top_k,
                enable_diversity=False  # Disable for fair comparison
            )

            if len(recs) == 0:
                continue

            recommended_items = set(recs['StockCode'].tolist())

            # Get actual items (from user history)
            user_history = recommender.get_user_history(user_id)
            if len(user_history) == 0:
                continue

            actual_items = set(user_history['StockCode'].tolist())

            # Calculate metrics
            hits = len(recommended_items & actual_items)
            precision = hits / len(recommended_items) if len(recommended_items) > 0 else 0
            recall = hits / len(actual_items) if len(actual_items) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        except Exception as e:
            continue

    # Average metrics
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    # F1 Score
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0

    return avg_precision, avg_recall, f1_score


def grid_search_weights(recommender, test_users, top_k=10):
    """
    Grid search untuk menemukan bobot optimal
    """
    print("Starting grid search for optimal weights...")
    print(f"Testing on {len(test_users)} users")
    print("="*70)

    # Define search space
    alpha_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    beta_values = [0.1, 0.2, 0.3, 0.4]

    results = []
    best_f1 = 0
    best_weights = None

    total_combinations = 0
    for alpha in alpha_values:
        for beta in beta_values:
            gamma = 1.0 - alpha - beta
            if gamma >= 0.1 and gamma <= 0.6:  # Valid range
                total_combinations += 1

    print(f"Total combinations to test: {total_combinations}\n")

    current = 0
    for alpha in alpha_values:
        for beta in beta_values:
            gamma = 1.0 - alpha - beta

            # Skip invalid combinations
            if gamma < 0.1 or gamma > 0.6:
                continue

            current += 1
            print(f"[{current}/{total_combinations}] Testing α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}...", end=" ")

            # Evaluate
            precision, recall, f1 = evaluate_weights(
                recommender, test_users, alpha, beta, gamma, top_k
            )

            print(f"F1={f1:.4f}")

            results.append({
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            # Track best
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (alpha, beta, gamma)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)

    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE!")
    print("="*70)
    print(f"\nBest weights: α={best_weights[0]:.1f}, β={best_weights[1]:.1f}, γ={best_weights[2]:.1f}")
    print(f"Best F1-Score: {best_f1:.4f}")
    print(f"\nTop 5 configurations:")
    print(results_df.head(5).to_string(index=False))

    return results_df, best_weights


if __name__ == "__main__":
    print("Loading recommender system...")
    recommender = HybridRecommender(models_dir='models', data_dir='data')

    # Get sample users for testing
    all_users = recommender.get_all_users()

    # Use 50 random users for faster testing
    np.random.seed(42)
    test_users = np.random.choice(all_users, size=min(50, len(all_users)), replace=False)

    print(f"Selected {len(test_users)} test users\n")

    # Run grid search
    results_df, best_weights = grid_search_weights(recommender, test_users, top_k=10)

    # Save results
    results_df.to_csv('models/weight_optimization_results.csv', index=False)
    print(f"\nResults saved to: models/weight_optimization_results.csv")

    # Save best weights
    best_config = {
        'alpha': best_weights[0],
        'beta': best_weights[1],
        'gamma': best_weights[2],
        'f1_score': results_df.iloc[0]['f1_score'],
        'precision': results_df.iloc[0]['precision'],
        'recall': results_df.iloc[0]['recall']
    }

    with open('models/optimal_weights.pkl', 'wb') as f:
        pickle.dump(best_config, f)

    print(f"Best configuration saved to: models/optimal_weights.pkl")
    print("\nDone!")
