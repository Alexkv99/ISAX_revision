import numpy as np
import time
import pandas as pd
import sys
from ISAX_tree import iSAXTree
from exact_search import ExactSearch

def generate_random_series(num_series, length=128):
    """Generates a list of random time series."""
    return [np.random.randn(length) for _ in range(num_series)]

def load_dataset(dataset_name):
    """Loads a real dataset based on user input."""
    if dataset_name == "Mallat":
        filepath = "Dataset/Mallat/Mallat_TEST.txt"
    elif dataset_name == "ECG":
        filepath = "Dataset/NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TEST.txt"
    else:
        print("Dataset not found.")
        sys.exit()

    print(f"Loading dataset: {dataset_name}")
    
    data = pd.read_csv(filepath, delim_whitespace=True, header=None, dtype=float, engine='python')
    data = data.dropna()  # Clean dataset
    
    labels = data.iloc[:, 0].values  # Extract labels
    time_series = data.iloc[:, 1:].values  # Extract time series
    
    # Normalize dataset
    mean_vals = np.mean(time_series, axis=1, keepdims=True)
    std_vals = np.std(time_series, axis=1, keepdims=True)
    std_vals[std_vals == 0] = 1  # Prevent division by zero
    time_series = (time_series - mean_vals) / std_vals

    return labels, time_series

def benchmark_insertion(tree, time_series_list):
    """Measures the time taken to insert time series into the iSAX tree."""
    start_time = time.time()
    for ts in time_series_list:
        tree.insert(ts)
    return time.time() - start_time

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
    num_series = 10000  # Number of time series to insert for random data
    query_count = 100   # Number of queries to test
    series_length = 128 # Length of each time series

    print("Generating random time series...")
    random_time_series = generate_random_series(num_series, series_length)
    random_queries = generate_random_series(query_count, series_length)

    dataset_name = str(sys.argv[3])  # Read dataset name from terminal
    _, dataset_time_series = load_dataset(dataset_name)
    
    # Limit dataset queries to same count as random queries
    dataset_queries = dataset_time_series[:query_count]

    # Initialize iSAX Trees
    print("\nBuilding iSAX Tree for Random Data...")
    random_tree = iSAXTree(word_length=8, alphabet_size=4, max_leaf_size=10)
    random_insertion_time = benchmark_insertion(random_tree, random_time_series)

    print("\nBuilding iSAX Tree for Dataset...")
    dataset_tree = iSAXTree(word_length=8, alphabet_size=4, max_leaf_size=10)
    dataset_insertion_time = benchmark_insertion(dataset_tree, dataset_time_series)

    print(f"\nInsertion Time - Random Data: {random_insertion_time:.4f} sec")
    print(f"Insertion Time - {dataset_name} Dataset: {dataset_insertion_time:.4f} sec")

    # Run Search Benchmarks
    print("\nRunning search benchmarks for Random Data...")
    random_approx_time, random_brute_time = benchmark_search(random_tree, random_queries, random_time_series)

    print("\nRunning search benchmarks for Dataset...")
    dataset_approx_time, dataset_brute_time = benchmark_search(dataset_tree, dataset_queries, dataset_time_series)

    # Display Results
    print("\n=== SEARCH PERFORMANCE ===")
    print(f"Random Data - Approximate Search: {random_approx_time:.4f} sec, Brute Force: {random_brute_time:.4f} sec")
    print(f"Dataset ({dataset_name}) - Approximate Search: {dataset_approx_time:.4f} sec, Brute Force: {dataset_brute_time:.4f} sec")

    print(f"\nSpeedup Factor (Random Data): {random_brute_time / random_approx_time:.2f}x")
    print(f"Speedup Factor ({dataset_name} Dataset): {dataset_brute_time / dataset_approx_time:.2f}x")

if __name__ == "__main__":
    main()
