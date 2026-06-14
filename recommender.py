"""
Sistem Rekomendasi Manufaktur - Core Logic
Menggabungkan CF + CBF + SVD untuk Hybrid Filtering
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import load_npz

class SparseLocIndexer:
    """Helper class to implement df.loc-like indexing for sparse matrices"""
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if row_key in self.wrapper.to_pos and col_key in self.wrapper.to_pos:
                r = self.wrapper.to_pos[row_key]
                c = self.wrapper.to_pos[col_key]
                return self.wrapper.sparse_matrix[r, c]
            return 0.0
        else:
            row_key = key
            if row_key in self.wrapper.to_pos:
                r = self.wrapper.to_pos[row_key]
                row_dense = self.wrapper.sparse_matrix[r].toarray()[0]
                return pd.Series(row_dense, index=self.wrapper.index_list)
            raise KeyError(f"Key {row_key} not found.")

class SparseMatrixWrapper:
    """Wrapper to make a sparse matrix behave like a pandas DataFrame for similarity matrix lookups"""
    def __init__(self, sparse_matrix, index_list):
        self.sparse_matrix = sparse_matrix
        self.index_list = index_list
        self.to_pos = {val: idx for idx, val in enumerate(index_list)}
        self.index = pd.Index(index_list)
        self.columns = pd.Index(index_list)
        self.loc = SparseLocIndexer(self)


class HybridRecommender:
    """Class untuk generate rekomendasi hybrid"""

    def __init__(self, models_dir='models', data_dir='data'):
        """Initialize recommender dengan semua model"""
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.load_all_models()

    def load_all_models(self):
        """Load CF, CBF, SVD models"""
        # User-item matrix (untuk CF dan filtering)
        self.user_item_matrix = pd.read_csv(
            os.path.join(self.data_dir, 'user_item_matrix.csv'),
            index_col=0
        )

        # Product info
        self.products = pd.read_csv(
            os.path.join(self.data_dir, 'products_cleaned.csv')
        )

        # CF model (Sparse user similarity)
        user_sim_sparse_path = os.path.join(self.models_dir, 'user_similarity_matrix_sparse.npz')
        user_sim_index_path = os.path.join(self.models_dir, 'user_similarity_matrix_sparse_index.pkl')
        if os.path.exists(user_sim_sparse_path) and os.path.exists(user_sim_index_path):
            try:
                user_sim_sparse = load_npz(user_sim_sparse_path)
                with open(user_sim_index_path, 'rb') as f:
                    user_sim_index_data = pickle.load(f)
                user_similarity_index = user_sim_index_data['index']
                self.user_similarity = SparseMatrixWrapper(user_sim_sparse, user_similarity_index)
                print("[OK] Loaded sparse user similarity matrix")
            except Exception as e:
                print(f"[WARNING] Failed to load sparse user similarity: {e}. Falling back to CSV.")
                self.user_similarity = pd.read_csv(
                    os.path.join(self.models_dir, 'user_similarity_matrix.csv'),
                    index_col=0
                )
        else:
            self.user_similarity = pd.read_csv(
                os.path.join(self.models_dir, 'user_similarity_matrix.csv'),
                index_col=0
            )

        # CBF models
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf = pickle.load(f)

        # Sparse product similarity
        prod_sim_sparse_path = os.path.join(self.models_dir, 'product_similarity_matrix_sparse.npz')
        prod_sim_index_path = os.path.join(self.models_dir, 'product_similarity_matrix_sparse_index.pkl')
        if os.path.exists(prod_sim_sparse_path) and os.path.exists(prod_sim_index_path):
            try:
                prod_sim_sparse = load_npz(prod_sim_sparse_path)
                with open(prod_sim_index_path, 'rb') as f:
                    prod_sim_index_data = pickle.load(f)
                product_similarity_index = prod_sim_index_data['index']
                self.product_similarity = SparseMatrixWrapper(prod_sim_sparse, product_similarity_index)
                print("[OK] Loaded sparse product similarity matrix")
            except Exception as e:
                print(f"[WARNING] Failed to load sparse product similarity: {e}. Falling back to CSV.")
                self.product_similarity = pd.read_csv(
                    os.path.join(self.models_dir, 'product_similarity_matrix.csv'),
                    index_col=0
                )
        else:
            self.product_similarity = pd.read_csv(
                os.path.join(self.models_dir, 'product_similarity_matrix.csv'),
                index_col=0
            )

        # SVD model
        with open(os.path.join(self.models_dir, 'svd_model.pkl'), 'rb') as f:
            self.svd_model = pickle.load(f)

        # User-item ratings (untuk SVD)
        self.user_item_ratings = pd.read_csv(
            os.path.join(self.data_dir, 'user_item_ratings.csv')
        )

        print("Semua model berhasil dimuat!")

    def get_cf_recommendations(self, user_id, k_neighbors=10, top_n=10):
        """Collaborative Filtering recommendations"""
        if user_id not in self.user_similarity.index:
            return pd.DataFrame(columns=['StockCode', 'CF_Score'])

        user_sim = self.user_similarity.loc[user_id].drop(user_id, errors='ignore')
        similar_users = user_sim.nlargest(k_neighbors).index

        user_purchased = self.user_item_matrix.loc[user_id]
        purchased_products = user_purchased[user_purchased > 0].index.tolist()

        product_scores = {}
        for sim_user in similar_users:
            sim_score = user_sim[sim_user]

            # Convert similar_user to int if user_item_matrix has int index
            sim_user_key = int(sim_user) if sim_user not in self.user_item_matrix.index else sim_user

            # Check if similar user exists in matrix
            if sim_user_key not in self.user_item_matrix.index:
                continue

            sim_products = self.user_item_matrix.loc[sim_user_key]
            sim_purchased = sim_products[sim_products > 0]

            for prod, qty in sim_purchased.items():
                if prod not in purchased_products:
                    product_scores[prod] = product_scores.get(prod, 0) + sim_score * qty

        if not product_scores:
            return pd.DataFrame(columns=['StockCode', 'CF_Score'])

        df = pd.DataFrame(list(product_scores.items()), columns=['StockCode', 'CF_Score'])
        return df.sort_values('CF_Score', ascending=False).head(top_n).reset_index(drop=True)

    def get_cbf_recommendations(self, user_id, top_n=10):
        """Content-Based Filtering recommendations"""
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=['StockCode', 'CBF_Score'])

        user_purchases = self.user_item_matrix.loc[user_id]
        purchased_products = user_purchases[user_purchases > 0]

        if len(purchased_products) == 0:
            return pd.DataFrame(columns=['StockCode', 'CBF_Score'])

        product_scores = {}
        for prod, qty in purchased_products.items():
            if prod not in self.product_similarity.index:
                continue

            sim_prods = self.product_similarity.loc[prod]
            for sim_prod, sim_score in sim_prods.items():
                if sim_prod not in purchased_products.index:
                    product_scores[sim_prod] = product_scores.get(sim_prod, 0) + sim_score * qty

        if not product_scores:
            return pd.DataFrame(columns=['StockCode', 'CBF_Score'])

        df = pd.DataFrame(list(product_scores.items()), columns=['StockCode', 'CBF_Score'])
        return df.sort_values('CBF_Score', ascending=False).head(top_n).reset_index(drop=True)

    def get_svd_recommendations(self, user_id, top_n=10):
        """SVD Matrix Factorization recommendations"""
        all_products = self.products['StockCode'].unique()
        user_purchases = self.user_item_ratings[
            self.user_item_ratings['user_id'] == user_id
        ]['item_id'].unique()

        products_to_predict = [p for p in all_products if p not in user_purchases]

        predictions = []
        for prod in products_to_predict:
            try:
                pred = self.svd_model.predict(user_id, prod)
                predictions.append({'StockCode': prod, 'SVD_Score': pred.est})
            except:
                continue

        if not predictions:
            return pd.DataFrame(columns=['StockCode', 'SVD_Score'])

        df = pd.DataFrame(predictions)
        return df.sort_values('SVD_Score', ascending=False).head(top_n).reset_index(drop=True)

    def normalize_scores(self, df, col_name):
        """Normalize scores to 0-1 range"""
        if len(df) == 0:
            return df

        min_val = df[col_name].min()
        max_val = df[col_name].max()

        if max_val == min_val:
            df[col_name] = 0.5
        else:
            df[col_name] = (df[col_name] - min_val) / (max_val - min_val)

        return df

    def get_popular_products(self, top_n=10):
        """Get most popular products (for cold start)"""
        popular = self.products.nlargest(top_n, 'TotalQuantitySold')[
            ['StockCode', 'Description', 'AvgPrice', 'TotalQuantitySold']
        ].copy()
        popular['Hybrid_Score'] = 1.0  # Max score for popular items
        popular['Reason'] = 'Popular product'
        return popular.reset_index(drop=True)

    def is_cold_start_user(self, user_id):
        """Check if user is new (cold start)"""
        if user_id not in self.user_item_matrix.index:
            return True
        user_purchases = self.user_item_matrix.loc[user_id]
        return (user_purchases > 0).sum() < 3  # Less than 3 purchases = cold start

    def get_hybrid_recommendations(self, user_id, alpha=0.4, beta=0.3, gamma=0.3, top_n=10,
                                   enable_diversity=True, diversity_threshold=0.7):
        """
        Generate Hybrid recommendations with cold start handling and explainability

        Parameters:
        - user_id: Customer ID
        - alpha: CF weight (default 0.4)
        - beta: CBF weight (default 0.3)
        - gamma: SVD weight (default 0.3)
        - top_n: Jumlah rekomendasi
        - enable_diversity: Enable diversity control
        - diversity_threshold: Similarity threshold for diversity (0-1)

        Returns:
        - DataFrame dengan rekomendasi hybrid + explanations
        """

        # COLD START HANDLING
        if self.is_cold_start_user(user_id):
            print(f"Cold start detected for user {user_id}. Using popular products.")
            return self.get_popular_products(top_n=top_n)

        # Get recommendations dari setiap algorithm
        cf_recs = self.get_cf_recommendations(user_id, top_n=top_n*2)  # Get more for diversity
        cbf_recs = self.get_cbf_recommendations(user_id, top_n=top_n*2)
        svd_recs = self.get_svd_recommendations(user_id, top_n=top_n*2)

        # Normalize scores
        cf_recs = self.normalize_scores(cf_recs.copy(), 'CF_Score')
        cbf_recs = self.normalize_scores(cbf_recs.copy(), 'CBF_Score')
        svd_recs = self.normalize_scores(svd_recs.copy(), 'SVD_Score')

        # Combine scores with source tracking
        hybrid_scores = {}
        sources = {}  # Track which algorithm contributed most

        for _, row in cf_recs.iterrows():
            prod = row['StockCode']
            score = alpha * row['CF_Score']
            hybrid_scores[prod] = hybrid_scores.get(prod, 0) + score
            if prod not in sources or score > sources[prod][1]:
                sources[prod] = ('CF', score)

        for _, row in cbf_recs.iterrows():
            prod = row['StockCode']
            score = beta * row['CBF_Score']
            hybrid_scores[prod] = hybrid_scores.get(prod, 0) + score
            if prod not in sources or score > sources[prod][1]:
                sources[prod] = ('CBF', score)

        for _, row in svd_recs.iterrows():
            prod = row['StockCode']
            score = gamma * row['SVD_Score']
            hybrid_scores[prod] = hybrid_scores.get(prod, 0) + score
            if prod not in sources or score > sources[prod][1]:
                sources[prod] = ('SVD', score)

        # Create final recommendations
        df = pd.DataFrame(list(hybrid_scores.items()), columns=['StockCode', 'Hybrid_Score'])
        df = df.sort_values('Hybrid_Score', ascending=False)

        # DIVERSITY CONTROL
        if enable_diversity and len(df) > 0:
            df = self.apply_diversity_filter(df, diversity_threshold, top_n)
        else:
            df = df.head(top_n)

        # Add product information
        df = df.merge(
            self.products[['StockCode', 'Description', 'AvgPrice']],
            on='StockCode',
            how='left'
        )

        # EXPLAINABILITY: Add reason column
        df['Reason'] = df['StockCode'].map(lambda x: self.get_recommendation_reason(sources.get(x, ('Hybrid', 0))[0]))

        return df.reset_index(drop=True)

    def get_recommendation_reason(self, source):
        """Generate explanation for recommendation"""
        reasons = {
            'CF': 'Customers like you also bought this',
            'CBF': 'Similar to products you purchased',
            'SVD': 'Based on your preferences',
            'Hybrid': 'Recommended for you'
        }
        return reasons.get(source, 'Recommended for you')

    def apply_diversity_filter(self, df, similarity_threshold, top_n):
        """
        Apply diversity control using Maximal Marginal Relevance (MMR)
        Remove products that are too similar to already selected ones
        """
        if len(df) == 0:
            return df

        selected = []
        remaining = df.to_dict('records')

        # Always add the top item
        if remaining:
            selected.append(remaining.pop(0))

        # Add items that are diverse enough
        while len(selected) < top_n and remaining:
            best_item = None
            best_score = -1

            for item in remaining:
                # Check similarity with already selected items
                max_similarity = 0
                item_code = item['StockCode']

                for sel_item in selected:
                    sel_code = sel_item['StockCode']

                    # Get similarity from product similarity matrix
                    if item_code in self.product_similarity.index and sel_code in self.product_similarity.columns:
                        similarity = self.product_similarity.loc[item_code, sel_code]
                        max_similarity = max(max_similarity, similarity)

                # MMR score: balance relevance and diversity
                mmr_score = item['Hybrid_Score'] - (0.5 * max_similarity)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = item

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                break

        return pd.DataFrame(selected)

    def get_user_history(self, user_id):
        """Get purchase history for a user"""
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()

        user_purchases = self.user_item_matrix.loc[user_id]
        purchased_products = user_purchases[user_purchases > 0]

        history = []
        for prod, qty in purchased_products.items():
            prod_info = self.products[self.products['StockCode'] == prod]
            if len(prod_info) > 0:
                history.append({
                    'StockCode': prod,
                    'Description': prod_info['Description'].values[0],
                    'AvgPrice': prod_info['AvgPrice'].values[0],
                    'Quantity': qty,
                    'TotalRevenue': prod_info['AvgPrice'].values[0] * qty
                })

        return pd.DataFrame(history).sort_values('Quantity', ascending=False)

    def get_all_users(self):
        """Get list of all users"""
        return sorted(self.user_item_matrix.index.tolist())

    def get_metrics_summary(self):
        """Get summary of all models' metrics"""
        try:
            with open(os.path.join(self.models_dir, 'cf_metrics_summary.csv')) as f:
                cf_metrics = pd.read_csv(f)

            with open(os.path.join(self.models_dir, 'cbf_metrics_summary.csv')) as f:
                cbf_metrics = pd.read_csv(f)

            with open(os.path.join(self.models_dir, 'svd_metrics_summary.csv')) as f:
                svd_metrics = pd.read_csv(f)

            return {
                'cf': cf_metrics.to_dict('records'),
                'cbf': cbf_metrics.to_dict('records'),
                'svd': svd_metrics.to_dict('records')
            }
        except:
            return None


if __name__ == "__main__":
    # Test recommender
    print("Testing Hybrid Recommender...")
    recommender = HybridRecommender(models_dir='models', data_dir='data')

    # Get sample user
    users = recommender.get_all_users()
    sample_user = users[0]

    print(f"\nTesting dengan customer {sample_user}:")

    # Get history
    history = recommender.get_user_history(sample_user)
    print(f"\nPurchase history ({len(history)} produk):")
    print(history.head())

    # Get recommendations
    recs = recommender.get_hybrid_recommendations(sample_user, top_n=10)
    print(f"\nTop 10 Hybrid Recommendations:")
    print(recs)
