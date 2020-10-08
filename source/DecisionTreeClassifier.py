import numpy as np
import bisect as bs
import helper_functions
from graphviz import Digraph

'''
Decision tree classifier implementation using
TDIDT (Top-Down Induction of Decision Trees) algorithm
based on information gain heuristic.
'''
class DecisionTreeClassifier:
    def __init__(self, max_depth, attributes, class_names):
        self.depth = 0
        self.max_depth = max_depth
        self.attributes = attributes
        self.class_names = class_names

    def fit(self, data, target):
        self.__fit(data, target, node={}, depth=0)

    def __fit(self, data, target, node, depth):
        if node is None or len(target) == 0:
            return None
        pos_cl_name = self.class_names[1]
        neg_cl_name = self.class_names[0]
        if np.all(target == target[0]): # monocromatic
            # make a leaf
            n_zeros = len(target) if target[0] == 0 else 0
            n_ones = len(target) - n_zeros
            node = {'class': target[0],
                    'samples': len(target),
                    neg_cl_name: n_zeros,
                    pos_cl_name: n_ones}
            self.root = node
            return node
        else:
            # find best split based on information gain
            col, spl_val, gain = self.__find_best_split(data, target)
            n_ones = np.sum(target)
            n_zeros = len(target) - n_ones
            cl = 1 if n_ones > n_zeros else 0
            if spl_val == None or depth == self.max_depth:
                # leaf
                node = {'class': target[0],
                        'samples': len(target),
                        neg_cl_name: n_zeros,
                        pos_cl_name: n_ones}
                self.root = node
                return node

            node = {'attr': self.attributes[col],
                    'index_col': col,
                    'split_value': spl_val,
                    'class': cl,
                    'samples': n_zeros + n_ones,
                    neg_cl_name: n_zeros,
                    pos_cl_name: n_ones}
            t_left = target[data[:, col] < spl_val]
            t_right = target[data[:, col] >= spl_val]
            node['left'] = self.__fit(data[data[:, col] < spl_val], t_left, {}, depth + 1)
            node['right'] = self.__fit(data[data[:, col] >= spl_val], t_right, {}, depth + 1)
            self.depth += 1
            self.root = node
            return node

    def __find_best_split(self, data, target):
        max_gain = 0
        col = None
        spl_val = None
        for i, c in enumerate(data.T):
            if np.all(c == c[0]): # monocromatic
                continue
            gain, cur_spl_val = self.__find_best_split_for_attr(c, target)
            assert cur_spl_val != None, 'Monocromatic data!'
            if gain > max_gain or spl_val == None:
                max_gain, col, spl_val = gain, i, cur_spl_val
        return col, spl_val, max_gain

    def __find_best_split_for_attr(self, col, target):
        # TODO: Handle categorcial attribute case.
        max_gain = 0
        spl_val = None
        new_arr = np.array([col, target])
        # sort the columns w.r.t. the first row values
        sarr = new_arr[:, new_arr[0].argsort()]
        # find places where the values of target (second) row changes from 0 to 1 / 1 to 0.
        indices = np.where(sarr[1, :-1] != sarr[1, 1:])[0]
        for i in indices:
            n_ones = np.sum(target)
            n_zeros = len(target) - n_ones
            # find the first index of sarr[0] with element > sarr[0, i]
            j = bs.bisect_right(sarr[0], sarr[0, i])
            if j > sarr.shape[1] - 1:
                continue
            median = (sarr[0, j - 1] + sarr[0, j]) / 2
            # number of 1 labels in the first part of the split
            n_ones_l = np.sum(sarr[1, :j])
            # number of 0 labels in the first part of the split
            n_zeros_l = j - n_ones_l
            gain = helper_functions.inform_gain(n_zeros_l, n_ones_l,
                                                n_zeros - n_zeros_l, n_ones - n_ones_l)
            if gain > max_gain or spl_val == None:
                max_gain, spl_val = gain, median
        return max_gain, spl_val

    def predict(self, data):
        results = np.zeros(data.shape[0])
        for i, row in enumerate(data):
            results[i] = self.__predict_sample(row)
        return results

    def __predict_sample(self, row):
        node = self.root
        while node.get('attr'):
            if row[node['index_col']] < node['split_value']:
                node = node['left']
            else:
                node = node['right']
        else:
            return node.get('class')

    def plot_tree(self, fname):
        graph = Digraph()
        helper_functions.make_digraph(self.root, graph, self.class_names)
        graph.render(fname, format='png', view=False)