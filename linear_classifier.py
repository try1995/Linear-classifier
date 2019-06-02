"""
    Loop:
    1.Sample a batch of data
    2.Forward prop it through the graph, get loss
    3.Backpropagation to calculate the gradients
    4.Update the parameters using the gradient
"""
from time import time
from LINEAR_CLASSIFIER_FOR_MINST.utils import *
start = time()

'1.Sample a batch of data'
images_matrix, labels_matrix = load_mnist(mnist_path)

'2.Forward prop it through the graph, get loss'
M = 60000           # 数据集行
N = 784             # 数据集列
step_size = 1e-4    # 学习率
train_times = 3000     # 迭代次数
parm = 1e-5          # weight regularization hyperparameter
lables = 10         # 预测标签数

D = np.random.randn(M, N)


'''数据集60000*784, W为784*10'''
W = np.random.randn(N, lables) / np.sqrt(N)
# W = np.random.randn(N, N) * 0.01
# W = np.random.randn(N, N) / np.sqrt(N/2)  #relu

# loss = svm_loss_many(images_matrix, labels_matrix, W)
# full_loss = full_loss(loss, W, parm)

'3.Backpropagation to calculate the gradients'

'4.Update the parameters using the gradient'
for i in range(train_times):
    mini_bath_data, mini_bath_lables = sample_training_data(images_matrix, labels_matrix, 256)
    weights_grad = svm_loss_gradient(mini_bath_data, mini_bath_lables, W, parm)
    W += -step_size * weights_grad

test_images_matrix, test_labels = load_mnist(mnist_path, kind='t10k')
accuracy, predict_labels = predict_pre(test_images_matrix[0:30], test_labels[0:30], W)
print(accuracy, predict_labels)
image_show_many(test_images_matrix[0:30], predict_labels)