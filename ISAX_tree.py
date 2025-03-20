import numpy as np
from collections import defaultdict

class PAA:
    """Piecewise Aggregate Approximation (PAA) for time series dimensionality reduction."""
    def __init__(self, segments):
        self.segments = segments

    def transform(self, ts):
        n = len(ts)
        segment_size = n // self.segments
        paa_representation = np.array([
            np.mean(ts[i * segment_size:(i + 1) * segment_size])
            for i in range(self.segments)
        ])
        return paa_representation

class SAX:
    """Symbolic Aggregate approXimation (SAX) encoding."""
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.breakpoints = self._generate_breakpoints()
    
    def _generate_breakpoints(self):
        """Generates breakpoints for SAX encoding using Gaussian distribution."""
        return np.round(np.percentile(np.random.randn(100000),
                                      np.linspace(0, 100, self.alphabet_size + 1)[1:-1]), 4)
    
    def transform(self, paa_rep):
        """Converts PAA representation to SAX symbols."""
        return np.array([np.sum(paa_value > self.breakpoints) for paa_value in paa_rep])

class iSAXNode:
    """Node in the iSAX tree."""
    def __init__(self, sax_word, depth=0, max_size=10, is_leaf= False):
        self.sax_word = sax_word  # SAX representation of this node
        self.children = {}  # Dictionary for child nodes
        self.time_series = []  # Store time series if leaf node
        self.depth = depth
        self.max_size = max_size  # Max number of time series in a leaf before splitting
        self.is_leaf = is_leaf
        
    def __lt__(self, other):
        """Define a comparison for heapq to work properly."""
        return False

    def insert(self, ts, sax_word):
        if len(self.time_series) < self.max_size:
            self.time_series.append((ts, sax_word))
        else:
            self.split()
            key = tuple(sax_word)
            if key not in self.children:
                self.children[key] = iSAXNode(sax_word, self.depth + 1, self.max_size)
            self.children[key].insert(ts, sax_word)

    def split(self):
        """Splits the node into child nodes by refining one dimension of SAX representation."""
        if not self.children:  # If not already split
            split_dim = self.depth % len(self.sax_word)  # Choose a dimension to refine
            for ts, sax_word in self.time_series:
                new_sax_word = sax_word.copy()
                new_sax_word[split_dim] += 1  # Increase granularity
                key = tuple(new_sax_word)
                if key not in self.children:
                    self.children[key] = iSAXNode(new_sax_word, self.depth + 1, self.max_size)
                self.children[key].insert(ts, new_sax_word)
            self.time_series = []  # Clear leaf node data

class iSAXTree:
    """iSAX Tree for indexing time series data."""
    def __init__(self, word_length=16, alphabet_size=8, max_leaf_size=10):
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.max_leaf_size = max_leaf_size
        self.paa = PAA(word_length)
        self.sax = SAX(alphabet_size)
        self.root = iSAXNode(sax_word=[0] * word_length, max_size=max_leaf_size)

    def insert(self, ts):
        """Insert a time series into the iSAX tree."""
        paa_rep = self.paa.transform(ts)
        sax_word = self.sax.transform(paa_rep)
        self.root.insert(ts, sax_word)

    def approximate_search(self, query_ts):
        """Find the closest approximate match for the query time series."""
        paa_rep = self.paa.transform(query_ts)
        sax_word = self.sax.transform(paa_rep)
        return self._traverse(self.root, sax_word)
    
    def _traverse(self, node, sax_word):
        """Recursive traversal to find the closest leaf node."""
        key = tuple(sax_word)
        if key in node.children:
            return self._traverse(node.children[key], sax_word)
        return node.time_series  # Return the time series in the closest leaf
