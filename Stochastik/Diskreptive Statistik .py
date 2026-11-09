# Ein Programm um Haeufigkeits Variabeln zu bestimmen
import numpy as np


omega = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
X = np.array([9, 8, 7, 7, 8, 6, 10, 6, 8, 7, 9, 11, 12, 7, 9, 12, 8, 7, 8, 9])
S = np.array([])

for i in range(len(X)):
    a= X[i]
    print("trial_i", i, ":", a)
    for j in range(len(S)+1):

        if len(S) == 0:
            np.append(S,a)
            print("S" , S)
        print("trial_j", j)

        if S[j] != a :
            np.append(S,a)
            print("trial :", j, ":", S[j])
        else:
            print("trial :", j, ":", S[j], "ist schon in S")

            

