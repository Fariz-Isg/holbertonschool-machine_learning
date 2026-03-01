#!/usr/bin/env python3
"""Isolation random tree implementation for anomaly detection."""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """Isolation random tree class."""

    def __init__(self, max_depth=10, seed=0, root=None):
        '''inittt'''
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        '''str doc'''
        return self.root.__str__() + "\n"

    def depth(self):
        '''depth tree'''
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''count dee'''
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        '''update tr'''
        self.root.update_bounds_below()

    def get_leaves(self):
        '''get the leaves'''
        return self.root.get_leaves_below()

    def update_predict(self):
        '''update tree predict function'''
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda x: np.sum(
            np.array([leaf.indicator(x) * leaf.value for leaf in leaves]),
            axis=0
        )

    def np_extrema(self, arr):
        '''np extrema'''
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        '''random split criterion'''
        sub_population = self.explanatory[node.sub_population]
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_values = sub_population[:, feature]
            min_val, max_val = self.np_extrema(feature_values)
            diff = max_val - min_val
        x = self.rng.uniform()
        threshold = (1 - x) * min_val + x * max_val
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        '''get a leaf child'''
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        '''get a node child'''
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        '''fit node'''
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        is_left_leaf = (node.depth + 1 >= self.max_depth) or (
            np.sum(left_population) <= self.min_pop
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 >= self.max_depth) or (
            np.sum(right_population) <= self.min_pop
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        '''fit the isolation random tree to the data'''
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
