#!/usr/bin/env python3
"""Random Forest"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Random Forest class"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """predict the value of a data point
        initialize an empty list to store the predictions of each tree"""
        predictions = []
        """iterate over the trees in the forest and get the
        predictions for each tree
        append the predictions to the tree_predictions list"""
        for predict_function in self.numpy_preds:
            predictions.append(predict_function(explanatory))

        predictions = np.array(predictions)

        mode_predictions = []
        for example_predictions in predictions.T:
            unique_values, counts = np.unique(example_predictions,
                                              return_counts=True)
            mode_index = np.argmax(counts)
            mode_predictions.append(unique_values[mode_index])

        return np.array(mode_predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """trains the random forest"""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")
            print(f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory,self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """Calculate the accuracy of the model on the test data."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size
