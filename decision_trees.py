# Ernesto Bernardo CS422 Machine Learning Project 1
# decision_trees.py

# import libraries
import numpy as np
import math


# Entropy
def H_whole(Y):
    # find num of Yes and No of whole set
    yes = 0
    no = 0
    i = 0
    while i < Y.size:
        if Y[i] == 0:
            no += 1
        else:
            yes += 1
        i += 1
    # calculate entropy
    n = no / Y.size
    y = yes / Y.size
    entropy = -n * math.log2(n) - y * math.log2(y)
    # print("Entropy_Whole:", entropy)
    return entropy


# entropy_feat, Finds the IG of each feature NOT already selected for DT
def H_feat(used, X, Y, entropy_whole):
    # use "used" array to track which features to calculate
    i = 0
    entropy_feat = []
    for x in used:
        # feat not used = find IG
        # print("--Feature", i, "--")
        if x == 0:
            # collect the numb of nos and yeses of feature i
            n_no = 0.0
            n_yes = 0.0
            n_total = 0.0
            y_no = 0.0
            y_yes = 0.0
            y_total = 0.0
            j = 0
            # for each label
            while j < Y.size:
                # if feature i = 0
                if X[j, i] == 0:
                    n_total += 1
                    # determine if feature's label outcome is no or yes
                    if Y[j] == 0:
                        n_no += 1
                    else:
                        n_yes += 1
                # if feature = 1
                else:
                    y_total += 1
                    # determine if feature's label outcome is no or yes
                    if Y[j] == 0:
                        y_no += 1
                    else:
                        y_yes += 1
                j += 1
            # calculate entropy of feature nos
            # if n or y = 0 or total = 0, entropy = 0
            if n_total == 0 or n_yes == 0 or n_no == 0:
                n_entropy = 0
            else:
                y = n_yes / n_total
                n = n_no / n_total
                n_entropy = - y * math.log2(y) - n * math.log2(n)
            # print("Entropy of F_no: ", n_entropy)
            # calculate entropy of feature yeses
            if y_total == 0 or y_yes == 0 or y_no == 0:
                y_entropy = 0
            else:
                y = y_yes / y_total
                n = y_no / y_total
                y_entropy = - y * math.log2(y) - n * math.log2(n)
            # print("Entropy of F_yes:", y_entropy)
            # calculate information gain
            IG = entropy_whole - (
                    (n_total / (n_total + y_total)) * n_entropy + (y_total / (n_total + y_total)) * y_entropy)
            # store IG in array, iterate i
            # print("IG:", IG)
            entropy_feat.append(IG)
        # if feat used, don't calculate IG, store -1
        else:
            entropy_feat.append(-1)
        i += 1
    # return IG in array
    entropy_feat = np.array(entropy_feat)
    return entropy_feat


# probability, used to figure out what choice to make for leaf nodes.
def probability(X, Y, feat):
    # collect the numb of nos and yeses of feature i
    n_no = 0.0
    n_yes = 0.0
    n_total = 0.0
    y_no = 0.0
    y_yes = 0.0
    y_total = 0.0
    j = 0
    prob = []
    # for each label
    while j < Y.size:
        # if feature i = 0
        if X[j, feat] == 0:
            n_total += 1
            # determine if feature's label outcome is no or yes
            if Y[j] == 0:
                n_no += 1
            else:
                n_yes += 1
        # if feature = 1
        else:
            y_total += 1
            # determine if feature's label outcome is no or yes
            if Y[j] == 0:
                y_no += 1
            else:
                y_yes += 1
        j += 1
    # calculate entropy of feature nos
    # if n or y = 0 or total = 0, entropy = 0
    if n_total == 0 or n_yes == 0 or n_no == 0:
        n_entropy = 0
    else:
        y = n_yes / n_total
        n = n_no / n_total
        n_entropy = - y * math.log2(y) - n * math.log2(n)
    # calculate entropy of feature yeses
    if y_total == 0 or y_yes == 0 or y_no == 0:
        y_entropy = 0
    else:
        y = y_yes / y_total
        n = y_no / y_total
        y_entropy = - y * math.log2(y) - n * math.log2(n)
    # calculate probability of n and of y
    if n_no / n_total >= n_yes / n_total:
        prob.append(0)
    else:
        prob.append(1)
    prob.append(n_entropy)
    if y_no / y_total >= y_yes / y_total:
        prob.append(0)
    else:
        prob.append(1)
    prob.append(y_entropy)
    prob = np.array(prob)
    return prob


# build_tree, calculates IG for each function and constructs tree recursively
def build_tree(used, X, Y, entropy_whole, max_depth):
    # find IG of all features
    entropy_feat = H_feat(used, X, Y, entropy_whole)

    # pick feature with best IG
    i = 0
    feat = -1
    max_ig = -1
    while i < X[0].size:
        if entropy_feat[i] > max_ig:
            max_ig = entropy_feat[i]
            feat = i
        i += 1

    # print("-----")
    # print("Feature:", feat)
    # print("IG:", max_ig)

    # Find prob of Y and N for feat 0 and 1, select best prob
    prob = probability(X, Y, feat)
    # print("prob", prob)

    # ---BASE CASES---
    # if IG of feat = H_whole, create leaf node
    if max_ig == entropy_whole:
        return [prob[0], prob[2], feat]

    # check if IG = 0
    if max_ig == 0:
        return [0, 0, feat]  # doesn't matter because IG is 0

    # if max depth == 1
    if max_depth == 1:
        return [prob[0], prob[2], feat]

    # change feat in used
    used[feat] = 1
    # if no more features
    if np.all(used == 1):
        return [prob[0], prob[2], feat]

    # ---NO BASE CASE---
    # determine what branch to make node and which to cont tree
    # [0 or 1, [...], feat]
    if prob[1] <= prob[3]:
        # take out samples that aren't being used in next iteration
        i = 0
        x_new = []
        y_new = []
        while i < Y.size:
            if X[i, feat] == 0:  # if data point's feature is no
                if Y[i] == prob[0]:  # and label is same as NO outcome
                    x_new = x_new
                    y_new = y_new
            else:  # else, add to new list
                x_new.append(X[i])
                y_new.append(Y[i])
            i += 1
        X = np.array(x_new)
        Y = np.array(y_new)
        # recursively call function
        DT = [prob[0], build_tree(used, X, Y, prob[3], max_depth - 1), feat]
    # [[...], 0 or 1, feat]
    else:
        i = 0
        x_new = []
        y_new = []
        while i < Y.size:
            if X[i, feat] == 1:  # if data point's feature is YES
                if Y[i] == prob[2]:  # and label is same as YES outcome
                    x_new = x_new  # take data point out of data set
                    y_new = y_new
            else:
                x_new.append(X[i])
                y_new.append(Y[i])
            i += 1
        X = np.array(x_new)
        Y = np.array(y_new)
        DT = [build_tree(used, X, Y, prob[1], max_depth - 1), prob[2], feat]

    return DT


# DT_train_binary(X, Y, max_depth)
def DT_train_binary(X, Y, max_depth):
    # invalid max_depth
    if max_depth == 0 or max_depth < -1:
        return None

    # size of X does not match size of Y
    if len(X) != Y.size:
        return None

    # data set is empty
    if X.size == 0 or Y.size == 0:
        return None

    # create a DT class
    # create temp array to track used features
    num_features = X[0].size
    i = 0
    used = []
    while i < num_features:
        used.append(0)
        i += 1
    used = np.array(used)

    # Search each feature & create tree recursively
    entropy_whole = H_whole(Y)
    DT = build_tree(used, X, Y, entropy_whole, max_depth)
    return DT


# DT_test_binary
def DT_test_binary(X, Y, DT):
    # take each sample, iterate through DT and find outcome
    # compare outcomes, keep track of # correct & total, return % acc
    correct = 0
    total = 0
    i = 0
    # for all samples
    while i < Y.size:
        # if no leaf visited, cont searching tree
        leaf = 0
        DT_temp = DT
        while leaf == 0:
            # if feat is 0, search left of tree
            if X[i, DT_temp[2]] == 0:
                # if DT is leaf, compare and increment numb
                if DT_temp[0] == 0 or DT_temp[0] == 1:
                    leaf = 1
                    # if prediction is same as sample, increment
                    if Y[i] == DT_temp[0]:
                        correct += 1
                        total += 1
                    else:
                        total += 1
                else:  # if DT is not a leaf, cont searching tree
                    DT_temp = DT_temp[0]
            else:  # if feat is 1, search right
                # if DT is leaf, compare and increment numb
                if DT_temp[1] == 0 or DT_temp[1] == 1:
                    leaf = 1
                    # if prediction is same as sample, increment
                    if Y[i] == DT_temp[1]:
                        correct += 1
                        total += 1
                    else:
                        total += 1
                else:  # if DT is not a leaf, cont searching tree
                    DT_temp = DT_temp[1]
        i += 1
    return correct / total


# DT_make_prediction
def DT_make_prediction(X, DT):
    # take sample features, iterate through DT and find outcome
    # return prediction of given sample
    prediction = -1
    # if no leaf visited, cont searching tree
    leaf = 0
    DT_temp = DT
    while leaf == 0:
        # if feat is 0, search left of tree
        if X[DT_temp[2]] == 0:
            # if DT is leaf, make prediction
            if DT_temp[0] == 0 or DT_temp[0] == 1:
                leaf = 1
                prediction = DT_temp[0]
            else:  # if DT is not a leaf, cont searching tree
                DT_temp = DT_temp[0]
        else:  # if feat is 1, search right
            # if DT is leaf, make prediction
            if DT_temp[1] == 0 or DT_temp[1] == 1:
                leaf = 1
                prediction = DT_temp[1]
            else:  # if DT is not a leaf, cont searching tree
                DT_temp = DT_temp[1]
    return prediction
