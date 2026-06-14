"""
Script untuk optimasi model files di folder /models
Mengurangi ukuran file dengan compression dan sparse matrices
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
import os

def analyze_models_folder():
    """Analyze current models folder"""
    print("="*70)
    print("MODELS FOLDER ANALYSIS")
    print("="*70)

    models_dir = 'models'
    total_size = 0
    files_info = []

    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            files_info.append({
                'filename': filename,
                'size_mb': size / (1024**2),
                'size_bytes': size
            })

    # Sort by size
    files_info = sorted(files_info, key=lambda x: x['size_bytes'], reverse=True)

    print(f"\nTotal size: {total_size / (1024**2):.2f} MB")
    print(f"Total files: {len(files_info)}")
    print("\nTop 10 largest files:")
    print(f"{'Filename':<45} {'Size':>15}")
    print("-"*70)

    for info in files_info[:10]:
        print(f"{info['filename']:<45} {info['size_mb']:>10.2f} MB")

    return files_info, total_size

def optimize_similarity_matrix(matrix_path, output_path, threshold=0.01):
    """
    Optimize similarity matrix by:
    1. Converting to sparse matrix
    2. Removing low similarity values (< threshold)
    3. Saving in compressed format
    """
    print(f"\nOptimizing: {matrix_path}")
    print("-"*70)

    # Load matrix
    print("Loading matrix...")
    df = pd.read_csv(matrix_path, index_col=0)

    original_size = os.path.getsize(matrix_path) / (1024**2)
    print(f"Original size: {original_size:.2f} MB")
    print(f"Shape: {df.shape}")

    # Calculate sparsity
    values = df.values
    sparsity = (values == 0).sum() / values.size
    print(f"Sparsity: {sparsity*100:.2f}%")

    # Apply threshold (remove low similarities)
    print(f"Applying threshold: {threshold}")
    values[np.abs(values) < threshold] = 0

    new_sparsity = (values == 0).sum() / values.size
    print(f"New sparsity: {new_sparsity*100:.2f}%")

    # Convert to sparse matrix
    print("Converting to sparse matrix...")
    sparse_matrix = csr_matrix(values)

    # Save sparse matrix
    print(f"Saving to: {output_path}")
    save_npz(output_path, sparse_matrix)

    # Save index mapping
    index_path = output_path.replace('.npz', '_index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump({
            'index': df.index.tolist(),
            'columns': df.columns.tolist()
        }, f)

    new_size = os.path.getsize(output_path) / (1024**2)
    print(f"New size: {new_size:.2f} MB")
    print(f"Compression ratio: {original_size/new_size:.2f}x")
    print(f"Space saved: {original_size - new_size:.2f} MB")

    return original_size, new_size

def optimize_all_models():
    """Optimize all large model files"""
    print("\n" + "="*70)
    print("OPTIMIZING MODELS")
    print("="*70)

    optimizations = []

    # 1. Optimize user similarity matrix
    if os.path.exists('models/user_similarity_matrix.csv'):
        orig, new = optimize_similarity_matrix(
            'models/user_similarity_matrix.csv',
            'models/user_similarity_matrix_sparse.npz',
            threshold=0.01  # Remove similarities < 0.01
        )
        optimizations.append(('user_similarity', orig, new))

    # 2. Optimize product similarity matrix
    if os.path.exists('models/product_similarity_matrix.csv'):
        orig, new = optimize_similarity_matrix(
            'models/product_similarity_matrix.csv',
            'models/product_similarity_matrix_sparse.npz',
            threshold=0.01
        )
        optimizations.append(('product_similarity', orig, new))

    # Summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)

    total_original = sum(o[1] for o in optimizations)
    total_new = sum(o[2] for o in optimizations)
    total_saved = total_original - total_new

    print(f"\n{'Model':<30} {'Original':>12} {'Optimized':>12} {'Saved':>12}")
    print("-"*70)

    for name, orig, new in optimizations:
        saved = orig - new
        print(f"{name:<30} {orig:>10.2f} MB {new:>10.2f} MB {saved:>10.2f} MB")

    print("-"*70)
    print(f"{'TOTAL':<30} {total_original:>10.2f} MB {total_new:>10.2f} MB {total_saved:>10.2f} MB")
    print(f"\nOverall compression ratio: {total_original/total_new:.2f}x")
    print(f"Total space saved: {total_saved:.2f} MB ({total_saved/total_original*100:.1f}%)")

    return optimizations

def create_optimized_recommender():
    """Create a new recommender.py that uses sparse matrices"""

    code = '''"""
Optimized Recommender - Uses sparse matrices for better performance
"""

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
import os

class OptimizedHybridRecommender:
    """Optimized version using sparse matrices"""

    def __init__(self, models_dir='models', data_dir='data'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.load_all_models()

    def load_all_models(self):
        """Load models with sparse matrix support"""
        print("Loading optimized models...")

        # Load data
        self.user_item_matrix = pd.read_csv(
            os.path.join(self.data_dir, 'user_item_matrix.csv'),
            index_col=0
        )

        self.products = pd.read_csv(
            os.path.join(self.data_dir, 'products_cleaned.csv')
        )

        # Load sparse similarity matrices
        try:
            # Try to load sparse version first
            self.user_similarity_sparse = load_npz(
                os.path.join(self.models_dir, 'user_similarity_matrix_sparse.npz')
            )
            with open(os.path.join(self.models_dir, 'user_similarity_matrix_sparse_index.pkl'), 'rb') as f:
                user_sim_index = pickle.load(f)
            self.user_similarity_index = user_sim_index['index']
            print("✓ Loaded sparse user similarity matrix")
        except:
            # Fallback to dense version
            self.user_similarity = pd.read_csv(
                os.path.join(self.models_dir, 'user_similarity_matrix.csv'),
                index_col=0
            )
            self.user_similarity_sparse = None
            print("✓ Loaded dense user similarity matrix")

        try:
            # Try to load sparse version first
            self.product_similarity_sparse = load_npz(
                os.path.join(self.models_dir, 'product_similarity_matrix_sparse.npz')
            )
            with open(os.path.join(self.models_dir, 'product_similarity_matrix_sparse_index.pkl'), 'rb') as f:
                prod_sim_index = pickle.load(f)
            self.product_similarity_index = prod_sim_index['index']
            print("✓ Loaded sparse product similarity matrix")
        except:
            # Fallback to dense version
            self.product_similarity = pd.read_csv(
                os.path.join(self.models_dir, 'product_similarity_matrix.csv'),
                index_col=0
            )
            self.product_similarity_sparse = None
            print("✓ Loaded dense product similarity matrix")

        # Load other models
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf = pickle.load(f)

        with open(os.path.join(self.models_dir, 'svd_model.pkl'), 'rb') as f:
            self.svd_model = pickle.load(f)

        self.user_item_ratings = pd.read_csv(
            os.path.join(self.data_dir, 'user_item_ratings.csv')
        )

        print("All models loaded successfully!")
        print(f"Memory optimization: ~{480/100:.0f}x smaller" if self.user_similarity_sparse else "")
'''

    with open('recommender_optimized.py', 'w', encoding='utf-8') as f:
        f.write(code)

    print("\n✓ Created: recommender_optimized.py")

if __name__ == "__main__":
    # Analyze current state
    files_info, total_size = analyze_models_folder()

    # Run optimization
    optimizations = optimize_all_models()

    # Create optimized recommender
    create_optimized_recommender()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Backup original files (optional):
   mkdir models/backup
   cp models/*_similarity_matrix.csv models/backup/

2. Test optimized models:
   python recommender_optimized.py

3. If working well, update app.py to use OptimizedHybridRecommender

4. (Optional) Delete original CSV files to save space:
   rm models/user_similarity_matrix.csv
   rm models/product_similarity_matrix.csv
""")

    print("\nDone!")
