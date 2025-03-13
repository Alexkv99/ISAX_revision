import numpy as np
import time
from ISAX_tree import iSAXTree

def generate_random_series(num_series, length=128):
    """Generates a list of random time series."""
    return [np.random.randn(length) for _ in range(num_series)]

def benchmark_insertion(tree, time_series_list):
    """Measures the time taken to insert time series into the iSAX tree."""
    start_time = time.time()
    for ts in time_series_list:
        tree.insert(ts)
    end_time = time.time()
    return end_time - start_time

def benchmark_search(tree, query_series, brute_force_list):
    """Measures the time taken for approximate search in the iSAX tree and compares it to brute force search."""
    
    # iSAX Approximate Search
    start_time = time.time()
    approx_results = [tree.approximate_search(query) for query in query_series]
    approx_time = time.time() - start_time
    
    # Brute Force Search
    start_time = time.time()
    brute_results = []
    for query in query_series:
        best_match = min(brute_force_list, key=lambda ts: np.linalg.norm(ts - query))
        brute_results.append(best_match)
    brute_time = time.time() - start_time
    
    return approx_time, brute_time

def main():
    num_series = 10000  # Number of time series to insert
    query_count = 100   # Number of queries to test
    series_length = 128 # Length of each time series
    
    print("Generating random time series...")
    time_series_list = generate_random_series(num_series, series_length)
    query_series = generate_random_series(query_count, series_length)
    
    print("Building iSAX Tree...")
    tree = iSAXTree(word_length=8, alphabet_size=4, max_leaf_size=10)
    insertion_time = benchmark_insertion(tree, time_series_list)
    
    print(f"Insertion Time: {insertion_time:.4f} seconds for {num_series} series")
    
    print("Running search benchmarks...")
    approx_time, brute_time = benchmark_search(tree, query_series, time_series_list)
    
    print(f"Approximate Search Time: {approx_time:.4f} seconds for {query_count} queries")
    print(f"Brute Force Search Time: {brute_time:.4f} seconds for {query_count} queries")
    print(f"Speedup Factor: {brute_time / approx_time:.2f}x")
    
if __name__ == "__main__":
    main()
