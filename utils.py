import os
import struct
import random
import numpy as np
import matplotlib.pyplot as plt

mnist_path = r"./MINST"

'''激活函数'''
act_fun = {"relu": lambda x: np.maximum(0, x), 'tanh': lambda x: np.tanh(x),
           'sigmoid': lambda x: 1 / (1 + np.exp(-x)), 'leaky_relu': lambda x: np.maximum(0.1 * x, x),
           'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1)}


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`
       训练数据集大小6000*784
    """
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        '去头，不可注释掉'
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        '去头，不可注释掉'
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def image_show_many(images_matrix, labels_matrix):
    # 显示0-9数字图片
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = images_matrix[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title(labels_matrix[i])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def image_show(images_matrix):
    # 显示单个，当该图片训练不对可以打开看看是什么鬼画符
    fig, ax = plt.subplots()
    img = images_matrix.reshape(28, 28)
    ax.imshow(img, cmap='Greys', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def svm_loss(x, y, w):
    # x一维行向量，y用一个整数表示标签，w权值矩阵
    scores = w.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins, axis=0)
    return loss_i


def svm_loss_scores(scores, y):
    # scores一维列向量，y用一个整数表示标签
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins, axis=0)
    return loss_i, margins.T[0]


def svm_loss_many(X, y, w, parm):
    """
    # x多维行向量，y用一维标签，w权值矩阵
    :param X: data_array
    :param y: data_labels
    :param w: weights
    :return: loss and scores
    """
    scores = X.dot(w)
    num_train = X.shape[0]
    correct_class_scores = scores[range(scores.shape[0]), y.tolist()].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(scores.shape[0]), list(y)] = 0
    loss = np.sum(margins) / num_train + parm*np.sum(w*w)
    return loss, margins, scores


def softmax_loss():
    pass


def full_loss(loss, w, x):
    ret = np.average(loss) + x * np.sum(pow(np.array(w), 2))
    return ret


def sample_training_data(data, lables, batch_size):
    # 从数据集中随机取数据，进行梯度计算
    # 一般取值32/64/128/256
    num_train = data.shape[0]
    batch_idx = np.random.choice(np.array(num_train), batch_size, replace=True)
    mini_bath_data = data[batch_idx]
    mini_bath_lables = lables[batch_idx]
    return mini_bath_data, mini_bath_lables


def svm_loss_gradient(X, y, w, reg):
    """
    referer_url
    https://blog.csdn.net/u012931582/article/details/57397535#mini-batch-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D
    :param X: data_array
    :param y: data_labels
    :param w: weights
    :param parm: 正则化系数
    :return: gradient
    """

    loss_i, margins, scores = svm_loss_many(X, y, w, reg)
    # print(loss_i)
    num_train = scores.shape[0]
    coeff_mat = np.zeros_like(scores)
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

    dW = (X.T).dot(coeff_mat)
    dW = dW / num_train + reg * w
    return dW


def predict_pre(images_matrix, labels, w):
    scores_labels = images_matrix.dot(w)
    predict_label = np.argmax(scores_labels, axis=1)
    return np.mean(predict_label == labels), predict_label


if __name__ == '__main__':
    images_matrix, labels_matrix = load_mnist(mnist_path)
    # print(type(images_matrix), labels_matrix)
    # image_show_many(images_matrix, labels_matrix)
    mini_bath_data, mini_bath_lables = sample_training_data(images_matrix, labels_matrix, 30)
    print(type(mini_bath_data), mini_bath_lables)
    image_show_many(mini_bath_data, mini_bath_lables)