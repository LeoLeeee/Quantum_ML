'''
Restricted Boltzmann Machine for fashion_mnist classfication.
'''
import tensorflow as tf
import numpy as np
from tqdm import tqdm
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = tf.reshape(train_images, (train_images.shape[0],train_images.shape[1]*train_images.shape[2]))
print(train_images.shape)
test_images = tf.reshape(test_images, (test_images.shape[0],test_images.shape[1]*test_images.shape[2]))
# binary the data
condition = tf.less(train_images/255, 0.5)
train_images_b = tf.where(condition, 0.0, 1.0).numpy()
condition = tf.less(test_images/255, 0.5)
test_images_b = tf.where(condition, 0.0, 1.0).numpy()

nv = 784  # number of the visible unit
nh = 10  # number of the hidden unit
lr = 0.01  # learning rate
Maxiter = 3600 # max loop time
batch_size = 100 
# initialize the parameter
w = tf.Variable(tf.random.normal((nh, nv), stddev=0.01))
b = tf.Variable(tf.random.normal((nv,), stddev=0.01))
c = tf.Variable(tf.random.normal((nh,), stddev=0.01))

# get batch, the batch is acting as a S_v
def get_random_batch(batch_size):
    index = np.random.randint(0, np.shape(train_images_b)[0], batch_size)
    return train_images_b[index]

def random_sample(phv):
    size = phv.shape[0]
    tmp = np.random.random_sample(size)
    h = phv>=tmp
    h=np.array(h).astype(np.float32)
    return h

def CDk_sampling(images, k):  # k>=2
    v1 = []
    h1 = []
    v2 = []
    h2 = []
    for v in images:
        v1 = np.append(v1,v)
        phv = tf.sigmoid(tf.reshape(tf.matmul(w,tf.reshape(v,(-1,1))),(-1,))+c)
        h = random_sample(phv)
        h1 = np.append(h1,h)
        for _ in range(k-1):
            pvh = tf.sigmoid(tf.reshape(tf.matmul(tf.reshape(h,(1,-1)),w),(-1,))+b)
            v = random_sample(pvh)
            phv = tf.sigmoid(tf.reshape(tf.matmul(w,tf.reshape(v,(-1,1))),(-1,))+c)
            h = random_sample(phv)
        v2 = np.append(v2, v)
        h2 = np.append(h2, h)
    v1 = v1.reshape(-1,nv).astype(np.float32)
    h1 = h1.reshape(-1,nh).astype(np.float32)
    v2 = v2.reshape(-1,nv).astype(np.float32)
    h2 = h2.reshape(-1,nh).astype(np.float32)
    return v1, v2, h1, h2

def CDk_classification(images):
    ypred = []
    for v in tqdm(images):
        phv = tf.sigmoid(tf.reshape(tf.matmul(w,tf.reshape(v,(-1,1))),(-1,))+c)
        ycls = np.where(phv>=np.max(phv),1.0,0.0)
        ypred = np.append(ypred, ycls)
    ypred = ypred.reshape(-1, nh)
    return ypred

def derivation(v1, v2, h1, h2):
    dw = 0.0
    db = 0.0
    dc = 0.0
    sumvphv = 0.0
    sumv = 0.0
    sumphv = 0.0
    for vk in v2:
        phvk = tf.sigmoid(tf.reshape(tf.matmul(w,tf.reshape(vk,(-1,1))),(-1,))+c)
        sumvphv = sumvphv + np.outer(phvk, vk)
        sumv = sumv + vk
        sumphv = sumphv + phvk
    sumvphv = sumvphv/np.float(batch_size)
    sumv = sumv/np.float(batch_size)
    sumphv = sumphv/np.float(batch_size)
    for vt in v1:
        phvt = tf.sigmoid(tf.reshape(tf.matmul(w,tf.reshape(vt,(-1,1))),(-1,))+c)
        dw = dw + np.outer(phvt,vt)-sumvphv
        db = db + vt-sumv
        dc = dc + phvt-sumphv
    return dw, db, dc

def main():
    global w, b, c
    for _ in tqdm(range(Maxiter)):
        train_data = get_random_batch(batch_size)
        v1, v2, h1, h2 = CDk_sampling(train_data, 3)  # v1:vt, v2: vk, h1:when vt, h2:when vk
        dw, db, dc = derivation(v1, v2, h1, h2)
        alpha = 0.0
        w = w + lr*(dw/float(batch_size)-alpha*w)
        b = b + lr*(db/float(batch_size)-alpha*b)
        c = c + lr*(dc/float(batch_size)-alpha*c)
    print(w,b,c)
    #test
    ypred = CDk_classification(test_images_b)  # h1 is the label
    labels = tf.one_hot(test_labels,10)  # cast number to one hot
    accuracy = np.sum(np.multiply(ypred, labels))/float(len(test_labels))
    print("The final accuracy is {:.4f}".format(accuracy))
    print(ypred[:5])
    print(labels[:5])

if __name__ == '__main__':
    main()

