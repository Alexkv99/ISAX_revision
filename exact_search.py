import numpy as np
import heapq
from ISAX_tree import iSAXTree

class ExactSearch:
    """Implements exact nearest neighbor search using a priority queue."""
    
    def __init__(self, tree):
        self.tree = tree
    
    def mindist(self, query_paa, node_sax):
        """Lower bounding distance between a query and an iSAX node."""
        return np.linalg.norm(query_paa - node_sax)
    
    def exact_search(self, query_ts):
        """Finds the exact nearest neighbor using a priority queue."""
        
        # Convert query time series to PAA and SAX representation
        query_paa = self.tree.paa.transform(query_ts)
        query_sax = self.tree.sax.transform(query_paa)
        
        # Priority queue for search (min-heap based on lower bound distance)
        pq = []
        heapq.heappush(pq, (0, self.tree.root))
        
        best_so_far = None
        best_dist = float('inf')
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            # If we reach a leaf node, compute exact distances
            if not node.children:
                for ts, _ in node.time_series:
                    dist = np.linalg.norm(query_ts - ts)
                    if dist < best_dist:
                        best_dist = dist
                        best_so_far = ts
            else:
                # Traverse child nodes in order of increasing lower bound distance
                for child in node.children.values():
                    child_dist = self.mindist(query_paa, child.sax_word)
                    if child_dist < best_dist:
                        heapq.heappush(pq, (child_dist, child))
        
        return best_so_far, best_dist

# Example Usage
def main():
    num_series = 10000  # Number of time series in the dataset
    query_count = 10     # Number of queries to test
    series_length = 128  # Length of each time series
    
    print("Generating random time series...")
    time_series_list = [np.random.randn(series_length) for _ in range(num_series)]
    query_series = [np.random.randn(series_length) for _ in range(query_count)]
    
    print("Building iSAX Tree...")
    tree = iSAXTree(word_length=8, alphabet_size=4, max_leaf_size=10)
    for ts in time_series_list:
        tree.insert(ts)
    
    search = ExactSearch(tree)
    
    print("Running exact search...")
    for query in query_series:
        nn, distance = search.exact_search(query)
        print(f"Exact Nearest Neighbor Found - Distance: {distance:.4f}")
    
if __name__ == "__main__":
    main()
