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
