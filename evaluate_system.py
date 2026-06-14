"""
Script untuk evaluasi lengkap sistem rekomendasi
Menghitung metrics: Precision, Recall, F1, Coverage, Diversity, Novelty
"""

import pandas as pd
import numpy as np
from recommender import HybridRecommender
import pickle
from collections import defaultdict

def calculate_coverage(all_recommendations, total_products):
    """Calculate catalog coverage"""
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)

    coverage = len(unique_recommended) / total_products
    return coverage

def calculate_diversity(all_recommendations, product_similarity_matrix):
    """Calculate average diversity of recommendations"""
    diversities = []

    for recs in all_recommendations:
        if len(recs) < 2:
            continue

        # Calculate pairwise dissimilarity
        dissimilarities = []
        for i, prod1 in enumerate(recs):
            for prod2 in recs[i+1:]:
                if prod1 in product_similarity_matrix.index and prod2 in product_similarity_matrix.columns:
                    similarity = product_similarity_matrix.loc[prod1, prod2]
                    dissimilarity = 1 - similarity
                    dissimilarities.append(dissimilarity)

        if dissimilarities:
            diversities.append(np.mean(dissimilarities))

    return np.mean(diversities) if diversities else 0

def calculate_novelty(all_recommendations, product_popularity):
    """Calculate average novelty (inverse popularity)"""
    novelties = []

    for recs in all_recommendations:
        for prod in recs:
            if prod in product_popularity:
                # Novelty = -log(popularity)
                popularity = product_popularity[prod]
                novelty = -np.log2(popularity + 1e-10)
                novelties.append(novelty)

    return np.mean(novelties) if novelties else 0

def evaluate_algorithm(recommender, test_users, algorithm='hybrid', top_k=10):
    """
    Evaluate a specific algorithm

    algorithm: 'cf', 'cbf', 'svd', 'hybrid'
    """
    print(f"\nEvaluating {algorithm.upper()} algorithm...")

    precisions = []
    recalls = []
    f1_scores = []
    all_recommendations = []

    for i, user_id in enumerate(test_users):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_users)} users")

        try:
            # Get recommendations based on algorithm
            if algorithm == 'cf':
                recs = recommender.get_cf_recommendations(user_id, top_n=top_k)
            elif algorithm == 'cbf':
                recs = recommender.get_cbf_recommendations(user_id, top_n=top_k)
            elif algorithm == 'svd':
                recs = recommender.get_svd_recommendations(user_id, top_n=top_k)
            elif algorithm == 'hybrid':
                recs = recommender.get_hybrid_recommendations(user_id, top_n=top_k, enable_diversity=False)
            else:
                continue

            if len(recs) == 0:
                continue

            recommended_items = set(recs['StockCode'].tolist())
            all_recommendations.append(list(recommended_items))

            # Get user history as ground truth
            user_history = recommender.get_user_history(user_id)
            if len(user_history) == 0:
                continue

            actual_items = set(user_history['StockCode'].tolist())

            # Calculate metrics
            hits = len(recommended_items & actual_items)
            precision = hits / len(recommended_items) if len(recommended_items) > 0 else 0
            recall = hits / len(actual_items) if len(actual_items) > 0 else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        except Exception as e:
            continue

    # Calculate aggregate metrics
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    # Calculate coverage
    total_products = len(recommender.products)
    coverage = calculate_coverage(all_recommendations, total_products)

    # Calculate diversity
    diversity = calculate_diversity(all_recommendations, recommender.product_similarity)

    # Calculate novelty
    product_popularity = recommender.products.set_index('StockCode')['TotalQuantitySold'].to_dict()
    total_sales = sum(product_popularity.values())
    product_popularity = {k: v/total_sales for k, v in product_popularity.items()}
    novelty = calculate_novelty(all_recommendations, product_popularity)

    results = {
        'Algorithm': algorithm.upper(),
        'Precision@10': avg_precision,
        'Recall@10': avg_recall,
        'F1-Score@10': avg_f1,
        'Coverage': coverage,
        'Diversity': diversity,
        'Novelty': novelty,
        'Num_Users_Evaluated': len(precisions)
    }

    return results

def run_full_evaluation(recommender, test_users, top_k=10):
    """Run evaluation on all algorithms"""
    print("="*70)
    print("FULL SYSTEM EVALUATION")
    print("="*70)
    print(f"Test users: {len(test_users)}")
    print(f"Top-K: {top_k}")

    results = []

    # Evaluate each algorithm
    for algo in ['cf', 'cbf', 'svd', 'hybrid']:
        result = evaluate_algorithm(recommender, test_users, algorithm=algo, top_k=top_k)
        results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))

    return results_df

if __name__ == "__main__":
    print("Loading recommender system...")
    recommender = HybridRecommender(models_dir='models', data_dir='data')

    # Get test users
    all_users = recommender.get_all_users()

    # Use 100 random users for evaluation
    np.random.seed(42)
    test_users = np.random.choice(all_users, size=min(100, len(all_users)), replace=False)

    print(f"Selected {len(test_users)} test users\n")

    # Run evaluation
    results_df = run_full_evaluation(recommender, test_users, top_k=10)

    # Save results
    results_df.to_csv('models/evaluation_results_complete.csv', index=False)
    print(f"\nResults saved to: models/evaluation_results_complete.csv")

    # Save detailed results
    with open('models/evaluation_detailed_complete.pkl', 'wb') as f:
        pickle.dump(results_df.to_dict('records'), f)

    print(f"Detailed results saved to: models/evaluation_detailed_complete.pkl")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    best_precision = results_df.loc[results_df['Precision@10'].idxmax()]
    best_recall = results_df.loc[results_df['Recall@10'].idxmax()]
    best_f1 = results_df.loc[results_df['F1-Score@10'].idxmax()]
    best_coverage = results_df.loc[results_df['Coverage'].idxmax()]
    best_diversity = results_df.loc[results_df['Diversity'].idxmax()]

    print(f"\nBest Precision@10: {best_precision['Algorithm']} ({best_precision['Precision@10']:.4f})")
    print(f"Best Recall@10: {best_recall['Algorithm']} ({best_recall['Recall@10']:.4f})")
    print(f"Best F1-Score@10: {best_f1['Algorithm']} ({best_f1['F1-Score@10']:.4f})")
    print(f"Best Coverage: {best_coverage['Algorithm']} ({best_coverage['Coverage']:.4f})")
    print(f"Best Diversity: {best_diversity['Algorithm']} ({best_diversity['Diversity']:.4f})")

    print("\nDone!")
