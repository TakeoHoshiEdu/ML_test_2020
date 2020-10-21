import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

answer = []
np.random.seed(0)
loop_max = 100
loop_list = list(range(0, loop_max))
a = MLPRegressor()

for loop in range(0, loop_max):
    n = (loop + 3)
    input_data = np.random.uniform(-0.5, 0.5, (n, 2))
    target = (input_data[:, 0] + input_data[:, 1]).reshape(n)
    a.fit(input_data, target)
    answer += [a.predict([[-0.3, 0.4]])]


plt.xlabel("Loop")
plt.ylabel("predicted value")
plt.plot(loop_list, answer, label='predicted value')
plt.legend(loc='best')
plt.savefig("ml_test_nn_result.png")

print("end")
