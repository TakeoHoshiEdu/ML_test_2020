from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def main():
    np.random.seed(0)
    n = 100
    loop_max = 100
    loop_list = list(range(0, loop_max))
    answer = []
    clf = Lasso(alpha = 0.001)
    
    for loop in range(0, loop_max):
        n = (loop + 3)
        input_data = np.random.uniform(-0.5, 0.5, (n, 2))
        target = (input_data[:, 0] + input_data[:, 1]).reshape(n)

        clf.fit(input_data, target)

        test_data = [[-0.3, 0.4]]
        answer += [clf.predict(test_data)]
    
    plt.xlabel("Loop")
    plt.ylabel("predicted value")
    plt.plot(loop_list, answer, label='predicted value')
    plt.legend(loc='best')
    plt.savefig("ml_test_lasso_result.png")   
    
if __name__ == '__main__':
    main()
    print("end")
