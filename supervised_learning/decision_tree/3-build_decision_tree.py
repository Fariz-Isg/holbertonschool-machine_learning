#!/usr/bin/env python3
'''decision tree implementation for classification and regression'''
import numpy as np


class Node:
    '''Node in the decision tree'''
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        '''initialize a node in the decision tree'''
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        '''returns the maximum depth of the tree below this node'''
        if self.is_leaf:
            return self.depth
        else:
            if self.left_child:
                left_depth = self.left_child.max_depth_below()
            else:
                left_depth = 0

            if self.right_child:
                right_depth = self.right_child.max_depth_below()
            else:
                right_depth = 0

            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        '''count the number of nodes below this node, including itself'''
        if self.is_leaf:
            return 1
        else:
            if self.left_child:
                left_count = self.left_child.count_nodes_below(
                    only_leaves=only_leaves)
            else:
                left_count = 0
            if self.right_child:
                right_count = self.right_child.count_nodes_below(
                    only_leaves=only_leaves)
            else:
                right_count = 0
            if only_leaves:
                return left_count + right_count
            else:
                return 1 + left_count + right_count

    def __str__(self):
        '''string representation of a node in the decision tree'''
        if self.is_leaf:
            return f"{self.left_child}\n"
        else:
            left_str = f'{self.left_child}' if self.left_child else ''
            right_str = f'{self.right_child}' if self.right_child else ''
            return (f"[feature={self.feature}, threshold={self.threshold}]\n" +
                    self.left_child_add_prefix(left_str) +
                    self.right_child_add_prefix(right_str))

    def left_child_add_prefix(self, text):
        '''lcap doc'''
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        '''rcap doc'''
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        '''get a list of all the leaf nodes below this node'''
        if self.is_leaf:
            return [self]
        else:
            leaves = []
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
            return leaves


class Leaf(Node):
    '''leaf node in the decision tree'''
    def __init__(self, value, depth=None):
        '''initialize a leaf node in the decision tree'''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        '''string representation of a leaf node'''
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        '''maximum depth below a leaf node is just the depth of the leaf itself
        '''
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''count the number of nodes below this node, including itself'''
        return 1

    def get_leaves_below(self):
        '''getting leaves'''
        return [self]


class Decision_Tree():
    '''Decision tree class for classification and regression'''
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''initialize the decision tree'''
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

    def __str__(self):
        '''string representation of the decision tree'''
        return self.root.__str__()

    def depth(self):
        '''depth of the tree is the maximum depth below the root node'''
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''count the number of nodes in the tree'''
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        '''getting leaves'''
        return self.root.get_leaves_below()
