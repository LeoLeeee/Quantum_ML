import pyqpanda.pyQPanda as pq
import numpy as np
import random
import os, struct
from array import array as pyarray
from numpy import array, int8, uint8, zeros
import matplotlib.pyplot as plt
import cv2
class NeuralNet(object):
    def __init__(self, sizes):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]
    def sigmoid(self, z):
        return 3.0 / (1.0 + np.exp(-z)) - 1.5
    def sigmoid_prime(self, z):
        return 3 / (1.0 + np.exp(-z)) * (1 - 1 / (1.0 + np.exp(-z)))
    def feedforward(self, x):
        for b, w in zip(self.b_, self.w_):
            x = self.sigmoid(np.dot(w, x) + b)
        return x
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.b_, self.w_):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers_):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.w_ = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.w_, nabla_w)]
        self.b_ = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b_, nabla_b)]
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                eva=self.evaluate(test_data)
                print("Epoch {0}: {1} / {2}={3}".format(j, eva, n_test, eva/float(n_test)))
            else:
                print("Epoch {0} complete".format(j))
    def evaluate(self, test_data):
        test_results = [(np.around(cal_circuit(self.feedforward(x))), y) for (x,y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    def save(self, file_name):
        np.savez(file_name, w=self.w_, b=self.b_)
    def load(self, file_name):
        f = np.load(file_name, allow_pickle=True)
        w = f['w']
        b = f['b']
        return w, b
def cal_circuit(thetas):
    pq.init(pq.QMachineType.CPU)
    qubitlist = pq.qAlloc_many(1)
    prog = pq.QProg()
    for theta, qubit in zip(thetas, qubitlist):
        prog.insert(pq.H(qubit))
        prog.insert(pq.RY(qubit, theta))
    result = pq.prob_run_dict(prog, qubitlist)
    states = []
    probabilities = []
    for key, val in result.items():
        states.append(int(key, 2))
        probabilities.append(val)
    states = np.array(states, 'float')
    probabilities = np.array(probabilities)
    expectation = np.sum(states * probabilities)
    pq.finalize()
    return np.array([expectation])
def gradient(x, y):
    z = cal_circuit(x)
    input_list = np.array(x)
    shift_right = input_list + np.ones(input_list.shape) * np.pi / 2
    shift_left = input_list - np.ones(input_list.shape) * np.pi / 2
    gradients = []
    for i in range(len(input_list)):
        expectation_right = cal_circuit([shift_right[i]])
        expectation_left = cal_circuit([shift_left[i]])
        gradient = expectation_right - expectation_left
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients * (z[0] - y)
def loss(z, y):
    error = z - y
    cost = error * error
    return cost
def train(y):
    thetas = np.random.randn(1)
    losses = []
    for i in range(300):
        z = cal_circuit(thetas)
        cost = loss(z[0], y)
        losses.append(cost)
        grad = gradient(thetas, y).reshape(1, -1)[0]
        thetas = thetas - 0.5 * grad
    print('iter{:3d} , loss = {:.4f}'.format(i, cost))
    return thetas
def load_mnist(dataset="training_data", digits=np.arange(2),path=".\\Mnist_by_benyuan\\MNIST_data"):
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images.idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")
    # print(os.path.curdir)
    # print(os.listdir(path="."))
    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels
THETAS = [[-1.455], [1.455]]
def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)
    X = [np.reshape(x, (28 * 28, 1)) for x in image]
    X = [x / 255.0 for x in X]
    if dataset == "training_data":
        Y = [THETAS[y[0]] for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')
def image_read(file_name):
    image = cv2.imread(file_name, 0)
    t, rst = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    x = rst.reshape(28*28, 1)
    return x
def predict(x, w_, b_):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()
    for b, w in zip(b_, w_):
        x = net.sigmoid(np.dot(w, x) + b)
    result = np.around(cal_circuit(x))
    print('预测值：', result)
if __name__ == '__main__':
    the = []
    for i in range(2):
        the.append(train(i))
    print("theta=",the)
    INPUT = 28 * 28
    OUTPUT = 1
    net = NeuralNet([INPUT, 40, OUTPUT])
    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')
    net.SGD(train_set, 30, 128, 1e-2, test_data=test_set)
    net.save('0_2_w_b_5.npz')
    w_, b_ = net.load('0_2_w_b_5.npz')
    fig, ax = plt.subplots(2, 5)
    for i in range(10):
        x = test_set[i][0]
        for b, w in zip(b_, w_):
            x = net.sigmoid(np.dot(w, x) + b)
        result = int(np.around(cal_circuit(x))[0])
        title = 'Predict: %d' % result
        ax[i//5][i%5].imshow(test_set[i][0].reshape(28,28), cmap = 'gray')
        ax[i//5][i%5].set_title(title)
        ax[i//5][i%5].set_xticks([])
        ax[i//5][i%5].set_yticks([])
    plt.tight_layout()
    plt.show()