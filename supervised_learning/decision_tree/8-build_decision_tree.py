#!/usr/bin/env python3
"""depth of a decision tree"""
import numpy as np


class Node:
    """
    Class that represents a decision tree node
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Class constructor for Node class
        Args:
            feature (_type_, optional): _description_. Defaults to None.
            threshold (_type_, optional): _description_. Defaults to None.
            left_child (_type_, optional): _description_. Defaults to None.
            right_child (_type_, optional): _description_. Defaults to None.
            is_root (bool, optional): _description_. Defaults to False.
            depth (int, optional): _description_. Defaults to 0.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Method that calculates the maximum depth of the current node
        Returns:
            int: maximum depth of the current node
        """
        # If the node is a leaf, its max depth is its own depth
        if not self.left_child and not self.right_child:
            return self.depth

        # Initialize depths assuming the current node is the deepest
        left_depth = self.depth
        right_depth = self.depth

        # Recursively find the max depth of the left subtree
        if self.left_child is not None:
            left_depth = self.left_child.max_depth_below()

        # Recursively find the max depth of the right subtree
        if self.right_child is not None:
            right_depth = self.right_child.max_depth_below()

        # Return the maximum of left and right depths
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Method that counts the number of nodes below the current node
        Args:
            only_leaves (bool, optional): _description_. Defaults to False.
        Returns:
            int: number of nodes below the current node
        """
        # If we are only counting leaves and this is not a leaf,
        # return count from children
        if only_leaves and not self.is_leaf:
            return (self.left_child.count_nodes_below(only_leaves=True)
                    if self.left_child else 0) + \
                (self.right_child.count_nodes_below(only_leaves=True)
                 if self.right_child else 0)

        # If we are counting all nodes, or this is a leaf node,
        # start with 1 (this node)
        count = 1 if not only_leaves or self.is_leaf else 0

        # Add counts from children if they exist
        count += self.left_child.count_nodes_below(
            only_leaves) if self.left_child else 0
        count += self.right_child.count_nodes_below(
            only_leaves) if self.right_child else 0

        return count

    def __str__(self):
        """
        Method that returns the string representation of the current node
        """
        # String representation for the current node
        node_str = (
            f"root [feature={self.feature}, threshold={self.threshold}]\n"
            if self.is_root else
            f"-> node [feature={self.feature}, "
            f"threshold={self.threshold}]\n"
        )

        # If the node is a leaf, simply return the string representation
        if self.is_leaf:
            return node_str

        # Formatting for the left and right children
        left_str = self.left_child_add_prefix(
            self.left_child.__str__()) if self.left_child else ""
        right_str = self.right_child_add_prefix(
            self.right_child.__str__()) if self.right_child else ""

        return node_str + left_str + right_str

    def left_child_add_prefix(self, text):
        """ Add prefix to the left child """
        lines = text.split("\n")
        # Adding prefix to the first line
        new_text = "    +--" + lines[0] + "\n"
        # Adding prefix to the rest of the lines
        new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
        # Append an additional newline character if there are multiple lines
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def right_child_add_prefix(self, text):
        """ Add prefix to the right child """
        lines = text.split("\n")
        # Adding prefix to the first line
        new_text = "    +--" + lines[0] + "\n"
        # Adding prefix to the rest of the lines
        new_text += "\n".join(["     " + "  " + line for line in lines[1:-1]])
        # Append an additional newline character if there are multiple lines
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def get_leaves_below(self):
        """ Method that returns the leaves below the current node """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            leaves += self.left_child.get_leaves_below()
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Update the bounds of the leaves below the current node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = max(
                        child.lower.get(self.feature, -np.inf), self.threshold)
                else:  # right child
                    child.upper[self.feature] = min(
                        child.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Update the indicator function for the leaves below
        the current node."""

        def is_large_enough(x):
            """Check if the input is large enough."""
            lower_bounds = np.array([self.lower.get(i, -np.inf)
                                     for i in range(x.shape[1])])
            return np.all(x >= lower_bounds, axis=1)

        def is_small_enough(x):
            """Check if the input is small enough."""
            upper_bounds = np.array([self.upper.get(i, np.inf)
                                     for i in range(x.shape[1])])
            return np.all(x <= upper_bounds, axis=1)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """ Method that predicts the value of a sample """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Class that represents a leaf node in a decision tree
    """

    def __init__(self, value, depth=None):
        """
        Class constructor for Leaf class
        Args:
            value (_type_): _description_.
            depth (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Method that calculates the maximum depth of the current node
        Returns:
            int: maximum depth of the current node
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Method that counts the number of nodes below the current node
        Args:
            only_leaves (bool, optional): _description_. Defaults to False.
        Returns:
            int: number of nodes below the current node
        """
        return 1

    def __str__(self):
        """
        Method that returns the string representation of the current node
        """
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """ Method that returns the leaves below the current node """
        return [self]

    def update_bounds_below(self):
        """ Method that updates the bounds of the decision tree """
        pass

    def pred(self, x):
        """ Method that predicts the value of a sample """
        return self.value


class Decision_Tree():
    """
    Class that represents a decision tree
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Class constructor for Decision_Tree class
        Args:
            max_depth (int, optional): _description_. Defaults to 10.
            min_pop (int, optional): _description_. Defaults to 1.
            seed (int, optional): _description_. Defaults to 0.
            split_criterion (str, optional): _description_.
                Defaults to "random".
            root (_type_, optional): _description_. Defaults to None.
        """
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
        """
        Method that calculates the depth of the decision tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Method that counts the number of nodes in the decision tree
        Args:
            only_leaves (bool, optional): _description_. Defaults to False.
        Returns:
            int: number of nodes in the decision tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Method that returns the string representation of the decision tree
        """
        return self.root.__str__()

    def get_leaves(self):
        """ Method that returns the leaves of the decision tree """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Method that updates the bounds of the decision tree """
        self.root.update_bounds_below()

    def update_indicator(self):
        """
        Updates the indicator functions for all nodes in the tree starting
        from the root.
        """
        self.root.update_indicator()

    def pred(self, x):
        """ Method that predicts the value of a sample """
        return self.root.pred(x)

    def update_predict(self):
        """ Method that updates the predict function """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def fit(self, explanatory, target, verbose=0):
        """split between random and Gini split criterion
        and fit the decision tree"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        # definde later
        self.fit_node(self.root)

        # defined in the previous task
        self.update_predict()

        if verbose == 1:
            # print("----------------------------------------------------")
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")
            # print("----------------------------------------------------")

    def np_extrema(self, arr):
        """returns the minimum and maximum of an array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """returns a random feature and threshold"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """fitting a node in the decision tree
        splitting between right and left children
        is left = feature > threshold
        is right = feature <= threshold"""
        node.feature, node.threshold = self.split_criterion(node)
        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population

        # Ensure that left_population and right_population
        # are the same length as self.target
        if len(left_population) != len(self.target):
            left_population = np.pad(left_population, (0, len(
                self.target) - len(left_population)),
                'constant', constant_values=(0))
        if len(right_population) != len(self.target):
            right_population = np.pad(right_population, (0, len(
                self.target) - len(right_population)),
                'constant', constant_values=(0))

        # Is left node a leaf ?
        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.depth = node.depth + 1
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.depth = node.depth + 1
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """returns a leaf node"""
        target_values = self.target[sub_population]
        values, counts = np.unique(target_values, return_counts=True)
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """creates a new node and returns it"""
        n = Node()
        n.depth = node.depth+1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """returns the accuracy of the decision tree on the test data"""
        preds = self.predict(test_explanatory) == test_target
        return np.sum(preds) / len(test_target)

    def possible_thresholds(self, node, feature):
        """
        Calculate possible thresholds for a feature in a node.

        Args:
            node (Node): Node to evaluate.
            feature (int): Feature index.

        Returns:
            numpy.ndarray: Midpoints between unique sorted feature values.
        """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Find the best threshold minimizing Gini impurity for a feature.

        Args:
            node (Node): Current node.
            feature (int): Feature index.

        Returns:
            numpy.ndarray: [best_threshold, min_gini].
        """
        thresholds = self.possible_thresholds(node, feature)
        indices = np.arange(self.explanatory.shape[0])[node.sub_population]
        feature_values = self.explanatory[indices, feature]
        target_reduced = self.target[indices]
        classes = np.unique(target_reduced)

        # Initialize gini sums
        gini_sums = []

        # Calculate Gini impurity for each threshold
        for threshold in thresholds:
            left_indices = feature_values > threshold
            right_indices = ~left_indices

            # Calculate gini for left and right splits
            gini_left, gini_right = 0, 0
            for cls in classes:
                p_left = np.mean(target_reduced[left_indices] == cls)
                p_right = np.mean(target_reduced[right_indices] == cls)
                gini_left += p_left * (1 - p_left)
                gini_right += p_right * (1 - p_right)

            # Calculate the weighted average Gini impurity
            left_size = np.sum(left_indices)
            right_size = np.sum(right_indices)
            total_size = left_size + right_size
            gini_sum = (
                (gini_left * left_size + gini_right * right_size) / total_size
            )
            gini_sums.append(gini_sum)

        # Find the minimum Gini sum and its corresponding threshold
        min_index = np.argmin(gini_sums)
        return np.array([thresholds[min_index], gini_sums[min_index]])

    def Gini_split_criterion(self, node):
        """
        Select the best feature and threshold based on Gini impurity.

        Args:
            node (Node): Node to split.

        Returns:
            tuple: (best_feature_index, best_threshold).
        """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
