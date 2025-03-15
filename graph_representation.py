import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd
from ISAX_tree import iSAXTree
from exact_search import ExactSearch
from main import load_mallat_data 
import sys


DATASET_PATHS = {
    "Mallat": "Dataset/Mallat/Mallat_TEST.txt",
    "ECG": "Dataset/NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TEST.txt",
}


def load_dataset(dataset_name):
    """Loads dataset using existing function from main.py"""
    if dataset_name not in DATASET_PATHS:
        print("Dataset not found")
        sys.exit()

    print(f"Loading dataset: {dataset_name}")
    labels, time_series = load_mallat_data(DATASET_PATHS[dataset_name])
    return labels, time_series


def plot_insertion_time(dataset_name):
    """Plots the time taken to insert time series into the iSAX tree."""
    labels, time_series = load_dataset(dataset_name)
    num_series_list = [1000, 3000, 5000, len(time_series)]  # Different sizes for testing
    times = []

    for num_series in num_series_list:
        tree = iSAXTree(word_length=8, alphabet_size=4, max_leaf_size=10)

        start_time = time.time()
        for ts in time_series[:num_series]:  # Use subsets of real data
            tree.insert(ts)
        times.append(time.time() - start_time)

    plt.figure(figsize=(8, 5))
    plt.plot(num_series_list, times, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Time Series")
    plt.ylabel("Insertion Time (seconds)")
    plt.title(f"iSAX Insertion Time ({dataset_name})")
    plt.grid(True)
    plt.show()


def plot_search_comparison(dataset_name):
    """Plots the time taken for approximate vs. exact search on real data."""
    labels, time_series = load_dataset(dataset_name)
    num_series = min(5000, len(time_series))  # Use a subset for performance
    query_count = 100

    tree = iSAXTree(word_length=8, alphabet_size=4, max_leaf_size=10)
    for ts in time_series[:num_series]:
        tree.insert(ts)

    exact_searcher = ExactSearch(tree)
    query_series = time_series[:query_count]

    start_time = time.time()
    approx_results = [tree.approximate_search(query) for query in query_series]
    approx_time = time.time() - start_time

    start_time = time.time()
    exact_results = [exact_searcher.exact_search(query) for query in query_series]
    exact_time = time.time() - start_time

    labels = ["Approximate Search", "Exact Search"]
    times = [approx_time, exact_time]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color=['blue', 'red'])
    plt.ylabel("Search Time (seconds)")
    plt.title(f"Approximate vs. Exact Search ({dataset_name})")
    plt.grid(axis="y")
    plt.show()


def plot_time_series_representation(dataset_name):
    """Plots a time series with its PAA and SAX representations using horizontal line segments like in the iSAX paper."""
    labels, time_series = load_dataset(dataset_name)
    ts = time_series[0]  # Select first time series for visualization

    tree = iSAXTree(word_length=8, alphabet_size=4, max_leaf_size=10)
    paa_rep = tree.paa.transform(ts)
    sax_rep = tree.sax.transform(paa_rep)

    segment_size = len(ts) // len(paa_rep)
    x_paa = np.arange(0, len(ts), segment_size)

    plt.figure(figsize=(10, 5))
    plt.plot(ts, label="Original Time Series", color='black', linewidth=1)

    paa_lines = []
    sax_lines = []

    # Plot PAA representation with horizontal lines
    for i in range(len(paa_rep)):
        line, = plt.plot([x_paa[i], x_paa[i] + segment_size], [paa_rep[i], paa_rep[i]], 
                         color='red', linewidth=2, linestyle="dashed")
        paa_lines.append(line)

    # Plot SAX representation with horizontal lines
    for i in range(len(sax_rep)):
        line, = plt.plot([x_paa[i], x_paa[i] + segment_size], [sax_rep[i], sax_rep[i]], 
                         color='blue', linewidth=2, linestyle="dotted")
        sax_lines.append(line)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Time Series, PAA, and SAX ({dataset_name})")
    
    # Manually add legend with correct labels
    plt.legend([plt.Line2D([0], [0], color='black', linewidth=1),
                plt.Line2D([0], [0], color='red', linewidth=2, linestyle="dashed"),
                plt.Line2D([0], [0], color='blue', linewidth=2, linestyle="dotted")],
               ["Original Time Series", "PAA", "SAX"])
    
    plt.grid(True)
    plt.show()



def main():
    dataset_name =  sys.argv[1]
    if dataset_name not in DATASET_PATHS:
        print("Dataset not found")
        sys.exit()

    plot_insertion_time(dataset_name)
    plot_search_comparison(dataset_name)
    plot_time_series_representation(dataset_name)


if __name__ == "__main__":
    main()
