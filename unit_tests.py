# Ernesto Bernardo CS491 DevOps Final Project
# unit_tests.py


import unittest
import numpy as np
import decision_trees as dt


class DecisionTreeUnitTests(unittest.TestCase):
    # unit test 1
    # --H_whole()--
    # conditional, include labels with both YESes and NOs
    def test_entropy_whole(self):
        # X = [[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]
        Y = [1, 1, 0]
        Y = np.array(Y)
        self.assertAlmostEqual(dt.H_whole(Y), 0.9183, 4)

    # --H_feat()--
    # conditional, unused feature, returns -1 for entropy_feat
    def test_unused_feat_entr(self):
        used = [1]
        X = [0]
        Y = [0]
        entropy_whole = 0.975

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)

        test_result = [-1]
        test_result = np.array(test_result)
        self.assertEqual(dt.H_feat(used, X, Y, entropy_whole), test_result)

    # conditional, have diff labels with diff feature values
    # [label 0, feature 0] = 0, label's feature is no
    # [label 1, feature 0] = 1, label's feature is yes
    def test_used_feat_entr(self):
        used = [0, 0]
        X = [[0, 1], [0, 1]]
        Y = [1, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0., 0.]
        test_result = np.array(test_result)
        # self.assertEqual(dt.H_feat(used, X, Y, entropy_whole).all(), test_result.all())
        self.assertTrue((dt.H_feat(used, X, Y, entropy_whole) == test_result).all())

    # conditional, have mult labels with same features values but diff label values
    # [label 0, feature 0] = 1, label 0 outcome 0 or no
    # [label 1, feature 0] = 1, label 1 outcome 1 or yes, both share same feature, but diff result

    # conditional, number of NOs = 0
    # every label has the feature as YES
    def test_feat_entr_all_yes(self):
        used = [0]
        X = [[1], [1], [1]]
        Y = [1, 1, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0.]
        test_result = np.array(test_result)
        # self.assertEqual(dt.H_feat(used, X, Y, entropy_whole).all(), test_result.all())
        self.assertTrue((dt.H_feat(used, X, Y, entropy_whole) == test_result).all())

    # conditional, number of YESes = 0
    # every label has the feature as NO
    def test_feat_entr_all_no(self):
        used = [0]
        X = [[0], [0], [0]]
        Y = [1, 1, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0.]
        test_result = np.array(test_result)
        # self.assertEqual(dt.H_feat(used, X, Y, entropy_whole).all(), test_result.all())
        self.assertTrue((dt.H_feat(used, X, Y, entropy_whole) == test_result).all())

    # --probability()--

    # feature has YES
    # feature has NO
    # labels of feature no have different label results
    # labels of feature yes have different label results
    def test_used_prob(self):
        feat = 0
        X = [[0], [0], [1], [1]]
        Y = [0, 1, 0, 1]

        X = np.array(X)
        Y = np.array(Y)
        # entropy_whole = dt.H_whole(Y)

        test_result = [0, 1, 0, 1]
        test_result = np.array(test_result)
        # self.assertEqual(dt.probability(X, Y, feat).all(), test_result.all())
        self.assertTrue((dt.probability(X, Y, feat) == test_result).all())

    # if feature has all yes labels (no = 0)
    def test_prob_all_yes(self):
        feat = 0
        X = [[0], [0], [1], [1]]
        Y = [1, 1, 1, 1]

        X = np.array(X)
        Y = np.array(Y)
        # entropy_whole = dt.H_whole(Y)

        test_result = [1, 0, 1, 0]
        test_result = np.array(test_result)
        # self.assertEqual(dt.probability(X, Y, feat).all(), test_result.all())
        self.assertTrue((dt.probability(X, Y, feat) == test_result).all())

    # if feature has all no labels (yes = 0)
    def test_prob_all_no(self):
        feat = 0
        X = [[0], [0], [1], [1]]
        Y = [0, 0, 0, 0]

        X = np.array(X)
        Y = np.array(Y)
        # entropy_whole = dt.H_whole(Y)

        test_result = [0, 0, 0, 0]
        test_result = np.array(test_result)
        # self.assertEqual(dt.probability(X, Y, feat).all(), test_result.all())
        self.assertTrue((dt.probability(X, Y, feat) == test_result).all())

    # probability feature NOs lNo > lYes
    # probability feature YESes lNo > lYES
    def test_prob_label_no(self):
        feat = 0
        X = [[0], [0], [0], [0], [1], [1], [1], [1], [1], [1]]
        Y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1]

        X = np.array(X)
        Y = np.array(Y)
        # entropy_whole = dt.H_whole(Y)

        test_result = [0, 0.8113, 0, 0.65]
        test_result = np.array(test_result)
        # self.assertEqual(dt.probability(X, Y, feat).all(), test_result.all())
        prob = dt.probability(X, Y, feat)

        self.assertEqual(prob[0], test_result[0])
        self.assertAlmostEqual(prob[1], test_result[1], 4)
        self.assertEqual(prob[2], test_result[2])
        self.assertAlmostEqual(prob[3], test_result[3], 4)

    # probability feature Nos lYes > lNo
    # probability feature YESes lYes > lNo
    def test_prob_label_yes(self):
        feat = 0
        X = [[0], [0], [0], [0], [1], [1], [1], [1], [1], [1]]
        Y = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0]

        X = np.array(X)
        Y = np.array(Y)
        # entropy_whole = dt.H_whole(Y)

        test_result = [1, 0.8113, 1, 0.6500]
        test_result = np.array(test_result)
        # self.assertEqual(dt.probability(X, Y, feat).all(), test_result.all())
        prob = dt.probability(X, Y, feat)

        self.assertEqual(prob[0], test_result[0])
        self.assertAlmostEqual(prob[1], test_result[1], 4)
        self.assertEqual(prob[2], test_result[2])
        self.assertAlmostEqual(prob[3], test_result[3], 4)

    # --build_tree()--
    # normal test :)
    # test with no edge cases?
    # condition, have the highest feature entropy in middle of list

    # base case 1: entropy_whole = max info gain
    def test_build_entropy_ig(self):
        used = [1, 1, 0]
        X = [[1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0, 1, 2]
        test_result = np.array(test_result)
        self.assertTrue((dt.build_tree(used, X, Y, entropy_whole, 2) == test_result).all())

    # base case 2: info gain = 0
    def test_build_ig_zero(self):
        used = [1, 1, 0]
        X = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0, 0, 2]
        test_result = np.array(test_result)
        self.assertTrue((dt.build_tree(used, X, Y, entropy_whole, 2) == test_result).all())

    # base case 3: max depth == 1
    def test_build_max_depth(self):
        used = [1, 1, 0]
        X = [[1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0, 1, 2]
        test_result = np.array(test_result)
        self.assertTrue((dt.build_tree(used, X, Y, entropy_whole, 1) == test_result).all())

    # base case 4: no more features
    def test_build_max_feat(self):
        used = [1, 1, 0]
        X = [[1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0, 1, 2]
        test_result = np.array(test_result)
        self.assertTrue((dt.build_tree(used, X, Y, entropy_whole, 2) == test_result).all())

    # no base case: NO entropy >= YES entropy
    # ~check data points
    def test_build_entropy_no(self):
        used = [0, 0, 0]
        X = [[0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0, 1, 0]
        test_result = np.array(test_result)
        DT = dt.build_tree(used, X, Y, entropy_whole, 2)
        self.assertEqual(DT[0], 0)
        self.assertTrue((DT[1] == test_result).all())
        self.assertEqual((DT[2]), 1)

    # no base case: YES entropy > NO entropy
    # ~check data points
    def test_build_entropy_yes(self):
        used = [0, 0, 0]
        X = [[0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]]
        Y = [1, 1, 0, 0]

        used = np.array(used)
        X = np.array(X)
        Y = np.array(Y)
        entropy_whole = dt.H_whole(Y)

        test_result = [0, 1, 0]
        test_result = np.array(test_result)
        DT = dt.build_tree(used, X, Y, entropy_whole, 2)
        self.assertTrue((DT[0] == test_result).all())
        self.assertEqual(DT[1], 0)
        self.assertEqual((DT[2]), 1)

    # --DT_train_binary()--
    # normal expected output
    def test_train_binary_tree(self):
        X = [[0, 1], [0, 0]]
        Y = [1, 0]

        X = np.array(X)
        Y = np.array(Y)

        test_result = [0, 1, 1]
        test_result = np.array(test_result)
        self.assertTrue((dt.DT_train_binary(X, Y, 2) == test_result).all())

    # invalid max_depth
    def test_train_invalid_depth(self):
        X = [[0, 1], [1, 0]]
        Y = [1, 0]

        X = np.array(X)
        Y = np.array(Y)
        self.assertEqual(dt.DT_train_binary(X, Y, 0), None)

    # invalid X and Y size
    def test_train_invalid_size(self):
        X = [[0, 1], [1, 0]]
        Y = [1, 0, 1]

        X = np.array(X)
        Y = np.array(Y)
        self.assertEqual(dt.DT_train_binary(X, Y, 2), None)

    # empty data set
    def test_train_empty_data(self):
        X = []
        Y = []

        X = np.array(X)
        Y = np.array(Y)
        self.assertEqual(dt.DT_train_binary(X, Y, 2), None)

    # --DT_test_binary()--
    # search tree left, find leaf
    def test_binary_left_leaf(self):
        X = [[0]]
        Y = [0]
        DT = [0, 1, 0]

        X = np.array(X)
        Y = np.array(Y)
        self.assertEqual(dt.DT_test_binary(X, Y, DT), 1.0)

    # search tree right, find leaf
    def test_binary_right_leaf(self):
        X = [[1]]
        Y = [1]
        DT = [0, 1, 0]

        X = np.array(X)
        Y = np.array(Y)
        self.assertEqual(dt.DT_test_binary(X, Y, DT), 1.0)

    # search tree left, no leaf, continue
    def test_binary_left_no_leaf(self):
        X = [[0, 1]]
        Y = [1]
        DT = [[0, 1, 1], 1, 0]

        X = np.array(X)
        Y = np.array(Y)
        self.assertEqual(dt.DT_test_binary(X, Y, DT), 1.0)

    # search tree right, no leaf, continue
    def test_binary_right_no_leaf(self):
        X = [[1, 0]]
        Y = [0]
        DT = [0, [0, 1, 1], 0]

        X = np.array(X)
        Y = np.array(Y)
        self.assertEqual(dt.DT_test_binary(X, Y, DT), 1.0)

    # --DT_make_prediction()--
    # search tree left, no leaf, continue
    # search tree left, find leaf, make prediction
    def test_prediction_left(self):
        X = [0, 0]
        DT = [[0, 1, 1], 1, 0]

        X = np.array(X)
        self.assertEqual(dt.DT_make_prediction(X, DT), 0.0)

    # search tree right, no leaf, continue
    # search tree right, find leaf, make prediction
    def test_prediction_right(self):
        X = [1, 1]
        DT = [0, [0, 1, 1], 0]

        X = np.array(X)
        self.assertEqual(dt.DT_make_prediction(X, DT), 1.0)


if __name__ == "__main__":
    unittest.main()
