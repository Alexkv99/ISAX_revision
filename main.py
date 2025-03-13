import numpy as np
import time
import pandas as pd
from ISAX_tree import iSAXTree
from exact_search import ExactSearch
import sys

DATASET_PATH = "Dataset/Mallat/Mallat_TEST.txt"

def load_mallat_data(filepath):
    """Loads and preprocesses the Mallat dataset."""
    print("Reading dataset...")
    
    # Auto-detect separator and ensure all values are numeric
    data = pd.read_csv(filepath, delim_whitespace=True, header=None, dtype=float, engine='python')
    
    # Drop rows with NaN values
    data = data.dropna()
    
    labels = data.iloc[:, 0].values  # First column is class label
    time_series = data.iloc[:, 1:].values  # Remaining columns are time series data
    
    print(f"Dataset shape after cleaning: {time_series.shape}")
    
    # Normalize the time series safely
    mean_vals = np.mean(time_series, axis=1, keepdims=True)
    std_vals = np.std(time_series, axis=1, keepdims=True)
    std_vals[std_vals == 0] = 1  # Avoid division by zero
    time_series = (time_series - mean_vals) / std_vals
    
    return labels, time_series

def compare_search_methods(tree, exact_searcher, query_series):
    """Compares approximate search vs. exact search in the iSAX tree."""
    
    # Approximate Search Timing
    start_time = time.time()
    approx_results = [tree.approximate_search(query) for query in query_series]
    approx_time = time.time() - start_time
    
    # Exact Search Timing
    start_time = time.time()
    exact_results = [exact_searcher.exact_search(query) for query in query_series]
    exact_time = time.time() - start_time
    
    return approx_results, approx_time, exact_results, exact_time

def main():
    print("Loading Mallat dataset...")
    labels, time_series = load_mallat_data(DATASET_PATH)
    num_series = len(time_series)
    # Number of queries to test (all)
    query_count = num_series
    
    print(f"Loaded {num_series} time series from Mallat dataset.")
    
    print("Building iSAX Tree...")
    # Read parameters from the terminal
    word_length = int(sys.argv[1])
    alphabet_size = int(sys.argv[2])
    tree = iSAXTree(word_length=word_length, alphabet_size=alphabet_size, max_leaf_size=10)
    for ts in time_series:
        tree.insert(ts)
    
    exact_searcher = ExactSearch(tree)
    
    print("Running search comparison...")
    approx_results, approx_time, exact_results, exact_time = compare_search_methods(tree, exact_searcher, time_series[:query_count])
    
    print(f"Approximate Search Time: {approx_time:.4f} seconds for {query_count} queries")
    print(f"Exact Search Time: {exact_time:.4f} seconds for {query_count} queries")
    print(f"Speedup Factor: {exact_time / approx_time:.2f}x")
    
    # Print a sample result comparison
    print("\nSample Query Comparison:")
    for i in range(min(3, len(approx_results))):
        if approx_results[i] and exact_results[i]:
            print(f"Query {i+1}:")
            print(f"  Approximate Match: {approx_results[i][0][0][:5]}... (SAX: {approx_results[i][0][1]})")
            print(f"  Exact Match: {exact_results[i][0][:5]}... (Distance: {exact_results[i][1]:.4f})")
            print("------------------------")
    # Average of distance between approximate and exact search for all queries
    avg_dist = np.mean([np.linalg.norm(approx_results[i][0][0] - exact_results[i][0]) for i in range(query_count)])
    print(f"\nAverage Distance Error: {avg_dist:.4f}")
if __name__ == "__main__":
    main()