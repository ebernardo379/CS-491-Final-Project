# Ernesto Bernardo CS491 DevOps Final Project
# integration_tests.py


import unittest
import numpy as np
import decision_trees as dt


class DecisionTreeIntegrationTests(unittest.TestCase):
    # Test integration of H_feat with build tree, return proper result
    def test_build_tree_H_feat(self):
        used = [0, 0, 0, 0]
        X = [[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 0]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        # build_tree should have picked feature with the highest IG as first feature
        ig = dt.H_feat(used, X, Y, entropy_whole)
        DT = dt.build_tree(used, X, Y, entropy_whole, 3)
        self.assertEqual(DT[2], ig.argmax())

    # Test integration of probability with build tree, return proper result
    def test_build_tree_prob(self):
        used = [0, 0, 0, 0]
        X = [[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 0]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        # build_tree should have picked feature with the lowest entropy as first feature
        prob = dt.probability(X, Y, 0)
        DT = dt.build_tree(used, X, Y, entropy_whole, 3)
        if prob[1] <= prob[3]:  # search left
            self.assertEqual(DT[0], prob[0])
        else:  # search right
            self.assertEqual(DT[1], prob[2])

    # test integration of H_whole() with DT_train_binary(), proper output
    # max_ig and entropy_whole match, first base case hit
    def test_train_entropy_whole(self):
        used = [1, 1, 0]
        X = [[1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)
        ig = dt.H_feat(used, X, Y, entropy_whole)
        # if entropy_whole and max_ig are the same
        if entropy_whole == ig[ig.argmax()]:
            # build_tree should return leaf node
            DT = dt.build_tree(used, X, Y, entropy_whole, 3)
            test_result = [0, 1, 2]
            test_result = np.array(test_result)
            self.assertTrue((dt.DT_train_binary(X, Y, 3) == test_result).all())

    # test build_tree() integration with DT_train_binary(), get proper result
    def test_train_build_tree(self):
        # built tree should result in leaf.
        used = [0, 0]
        X = [[0, 1], [0, 0]]
        Y = [1, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = dt.build_tree(used, X, Y, entropy_whole, 2)
        self.assertTrue((dt.DT_train_binary(X, Y, 2) == test_result))

    # test DT_train_binary() and DT_test_binary() with DT_train_and_test()
    def test_train_and_test(self):
        X = [[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]
        Y = [1, 1, 0]
        X = np.array(X)
        Y = np.array(Y)

        max_depth = 3
        DT_result = dt.DT_train_binary(X, Y, max_depth)
        test_acc_result = dt.DT_test_binary(X, Y, DT_result)

        # ensure DT_train_and_test_binary is returning the proper result
        test_acc, DT = dt.DT_train_and_test_binary(X, Y, X, Y, max_depth)

        self.assertEqual(test_acc, test_acc_result)
        self.assertTrue(DT == DT_result)
