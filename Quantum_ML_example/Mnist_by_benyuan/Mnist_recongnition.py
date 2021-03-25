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
        # print("sizes=",sizes,sizes[:-1], sizes[1:],sizes[1:])
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]
        # for item in self.w_:
        #     print(item.shape)
        # for item in self.b_:
        #     print(item.shape)
    def sigmoid(self, z):
        return 5/ (1.0 + np.exp(-z)) - 2.5
    def sigmoid_prime(self, z):
        return 5 / (1.0 + np.exp(-z)) * (1 - 1 / (1.0 + np.exp(-z)))
  
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
            # print(np.dot(w, activation).shape,b.shape,z.shape)
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # print(activations[-1], y,self.sigmoid_prime(zs[-1]))
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers_):
            # print("L=",-l,-l+1,-l-1,len(self.w_),len(activations),len(activations[0]))
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
        self.w_ = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.w_,
        nabla_w)]
        self.b_ = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b_,
        nabla_b)]
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                self.save('0_3_w_b_5.npz')
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
        self.w_ = w
        b = f['b']
        self.b_ = b
        return w, b
def cal_circuit(thetas):
    pq.init(pq.QMachineType.CPU)
    qubitlist = pq.qAlloc_many(4)
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
    # print(states,probabilities)
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
    thetas = np.random.randn(4)
    losses = []
    cost = 1
    while cost>1e-5:
        z = cal_circuit(thetas)
        cost = loss(z[0], y)
        losses.append(cost)
        grad = gradient(thetas, y).reshape(1, -1)[0]
        thetas = thetas - 0.1 * grad
    print('iter{:3d} , loss = {:.4f}'.format(y, cost))
    thetas_new = []
    for the in thetas:
        while the < -np.pi:
            the = the+np.pi*2
        while the > np.pi:
            the = the - np.pi*2
        thetas_new.append(the)
    return thetas_new
def load_mnist(dataset="training_data", digits=np.arange(9), path=".\\Mnist_by_benyuan\\MNIST_data"):
    print(os.path.curdir)
    print(os.listdir(path="."))
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images.idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")
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
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows *cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels

THETAS = [array([-1.55044938, -7.82538107, -7.88372698, -1.60183282]), array([-0.6828993 , -0.88574621, -1.62720529, -2.11927666]), array([-2.51570566,  5.28390777, -1.19745135, -0.67634472]), array([-2.12856604,  0.23838136, -1.47770668, -2.52268882]), array([-1.45071479, -1.45763428,  1.37894139, -1.68640071]), array([-1.07716058, -0.03697881,  1.41473985, -1.59892726]), array([-0.75078514, -0.45396622,  1.32476298, -0.72753812]), array([ 0.59370098, -0.27811644,  1.31208441, -0.65562543]), array([ 2.42667994,  1.93975293,  1.64574306, -0.75971712]), array([ 3.12853333,  0.83431772, -1.54666783,  0.75874673])]

def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)
    X = [np.reshape(x, (28 * 28, 1)) for x in image]
    X = [x / 255.0 for x in X]
    if dataset == "training_data":
        Y=[]
        for y in label:
            Y.append([[THETAS[y[0]][0]],[THETAS[y[0]][1]],[THETAS[y[0]][2]],[THETAS[y[0]][3]]])
        print("lenX,Y=",len(X),len(Y))
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
    # the = []
    # for i in range(10):
    #     the.append(train(i))
    # print("theta=",the)
    
    THETAS = [[-1.5312615266418361, -1.5430713199165214, -1.5453468188530282, -1.5413362609834276], [-1.5572229811469687, -1.3822182851006513, -0.5325997958101794, -1.5824932673055818], [-1.213913103140634, -2.665079591672047, -0.7652042968256841, -0.9214998450799561], [-1.4293889163789986, -0.9532571786487305, 0.3786817048183099, -1.3795467315841687], [-0.8068002096497555, 0.5596830035354999, 0.03438464942494177, -1.2062861293497062], [-0.12379756819953969, -1.4529886076025356, -1.023453733568457, 0.06600873414367954], [1.5748323881309283, 1.1872476134487016, -0.705162453897398, -0.4203071575304124], [-1.3521050141135422, -0.8587413237349895, 0.4018511844032649, -0.009125258525921805], [1.4600320301859755, 1.4846406036633406, 1.102707724623792, -2.373128012413683], [1.2271032817996832, -1.7285932898746605, 1.2955083935387341, 0.022723036677278263]]
    print(THETAS)
    for i, theta in enumerate(THETAS):
        print("label, result",(i,cal_circuit(theta)))
    INPUT = 28 * 28
    OUTPUT = 4
    net = NeuralNet([INPUT, 64, OUTPUT])
    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')
    net.load('0_3_w_b_5.npz')
    net.SGD(train_set, 5000,  512, 1e-2, test_data=test_set)
    net.save('0_3_w_b_5.npz')
    w_, b_ = net.load('0_3_w_b_5.npz')
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

    ## shell 1