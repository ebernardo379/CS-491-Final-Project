import numpy as np
import decision_trees as dt

X = [[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]
Y = [1, 1, 0]
X = np.array(X)
Y = np.array(Y)

max_depth = 3
DT = dt.DT_train_binary(X,Y,max_depth)
test_acc = dt.DT_test_binary(X,Y,DT)
print("DT:",test_acc)