#!/usr/bin/env python3
"""Module for building a decision tree."""
import numpy as np


class Node:
    """Represents an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth among all nodes below this node."""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Return the number of nodes below this node."""
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)
        return left + right if only_leaves else 1 + left + right

    def left_child_add_prefix(self, text):
        """Add the left-child prefix to each line of text."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add the right-child prefix to each line of text."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return a string representation of the node and its subtree."""
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            text = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"
        text += self.left_child_add_prefix(self.left_child.__str__())
        text += self.right_child_add_prefix(self.right_child.__str__())
        return text.rstrip("\n")

    def get_leaves_below(self):
        """Return the list of all leaves below this node."""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively compute and attach lower/upper bound dicts to nodes."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Compute and store the indicator function from lower/upper bounds."""
        def is_large_enough(x):
            """Return True for each individual whose features exceed lower."""
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower]), axis=0)

        def is_small_enough(x):
            """Return True for each individual whose features are <= upper."""
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Predict the class of a single individual by traversing the tree."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Represents a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 since a leaf is always counted."""
        return 1

    def __str__(self):
        """Return a string representation of the leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return this leaf as a single-element list."""
        return [self]

    def update_bounds_below(self):
        """No children to update bounds for."""
        pass

    def pred(self, x):
        """Return the leaf value as the prediction."""
        return self.value


class Decision_Tree():
    """Represents a decision tree classifier."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the decision tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the decision tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return the list of all leaves in the decision tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the bounds of all nodes in the decision tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Compute and store the vectorised predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array(
            [leaves[np.argmax(
                np.array([leaf.indicator(A)[i] for leaf in leaves])
            )].value for i in range(A.shape[0])])

    def pred(self, x):
        """Predict the class of a single individual by traversing the tree."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return the min and max of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Return a random feature and threshold to split a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def possible_thresholds(self, node, feature):
        """Return midpoints between consecutive unique feature values."""
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Return best threshold and Gini score for a single feature."""
        thresholds = self.possible_thresholds(node, feature)
        sub_features = self.explanatory[:, feature][node.sub_population]
        sub_target = self.target[node.sub_population]
        classes = np.unique(sub_target)
        n = sub_features.size

        # goes_left[i, j]: individual i goes left for threshold j  → (n, t)
        goes_left = sub_features[:, np.newaxis] > thresholds[np.newaxis, :]

        # class_membership[i, k]: individual i belongs to class k  → (n, c)
        class_membership = (
            sub_target[:, np.newaxis] == classes[np.newaxis, :])

        # left_class_counts[j, k]: left-child count of class k at threshold j
        left_class_counts = (
            goes_left[:, :, np.newaxis] &
            class_membership[:, np.newaxis, :]
        ).sum(axis=0)                                          # (t, c)

        left_total = goes_left.sum(axis=0)                    # (t,)
        right_total = n - left_total                          # (t,)

        total_class_counts = class_membership.sum(axis=0)    # (c,)
        right_class_counts = (
            total_class_counts[np.newaxis, :] - left_class_counts)  # (t, c)

        # Gini impurities; guard against zero-size children
        left_fracs = left_class_counts / np.maximum(
            left_total[:, np.newaxis], 1)
        right_fracs = right_class_counts / np.maximum(
            right_total[:, np.newaxis], 1)

        gini_left = 1 - np.sum(left_fracs ** 2, axis=1)     # (t,)
        gini_right = 1 - np.sum(right_fracs ** 2, axis=1)   # (t,)

        gini_split = (
            left_total * gini_left + right_total * gini_right) / n  # (t,)

        best = np.argmin(gini_split)
        return thresholds[best], gini_split[best]

    def Gini_split_criterion(self, node):
        """Return the feature and threshold that minimise the Gini split."""
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree on the given data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(self.explanatory, self.target)    }""")

    def fit_node(self, node):
        """Recursively fit a node by choosing a split and creating children."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] > node.threshold))
        right_population = (
            node.sub_population &
            ~(self.explanatory[:, node.feature] > node.threshold))

        is_left_leaf = (
            np.sum(left_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create and return a leaf child with the majority class value."""
        classes, counts = np.unique(
            self.target[sub_population], return_counts=True)
        value = classes[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create and return an internal node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Return the accuracy of predictions on the given data."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
