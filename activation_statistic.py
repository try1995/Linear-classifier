import numpy as np
import matplotlib.pyplot as plt
from time import time
'''激活函数策略和权值的选择'''

start = time()
D = np.random.randn(1000, 500)
hidden_layer_sizes = [500]*10
nonlinearities = ['relu']*len(hidden_layer_sizes)

'''激活函数'''
act = {"relu": lambda x: np.maximum(0, x), 'tanh': lambda x: np.tanh(x),
       'sigmoid': lambda x: 1/(1+np.exp(-x)), 'leaky_relu': lambda x: np.maximum(0.1*x, x),
       'elu': lambda x: np.where(x > 0, x, np.exp(x)-1)}

Hs = {}
for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i-1]
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    '''不同的权重'''
    W = np.random.randn(fan_in, fan_in) / np.sqrt(fan_in)
    # W = np.random.randn(fan_in, fan_in) * 0.01
    # W = np.random.randn(fan_in, fan_in) / np.sqrt(fan_in/2)

    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    Hs[i] = H

print('input layer had mean %s and std %s' % (np.mean(D), np.std(D)))
layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]
for i, H in Hs.items():
    print('hidden layer %s had mean %s and std %s' % (i+1, layer_means[i], layer_stds[i]))

'''图形展示'''
plt.figure()
plt.subplot(121)
plt.plot(Hs.keys(), layer_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(Hs.keys(), layer_stds, 'or-')
plt.title('layer std')

plt.figure()
for i, H in Hs.items():
    plt.subplot(1, len(Hs), i+1)
    plt.hist(H.ravel(), 30, range=(-1, 1))

end = time()
print(end-start)
plt.show()
