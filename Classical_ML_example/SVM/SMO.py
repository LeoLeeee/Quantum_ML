'''
SMO algorithm for solving Support Vector Machine
'''
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

def generate_data(w=1.0, b=0.3):
    if b<0:
        b=-b
    elif b==0:
        b=1
    temp = np.random.random_sample(100)
    x = []
    y = []
    for x1 in temp:
        yr = np.random.random_sample(1)
        if yr > 0.5:
            x2 = w*x1+b + np.random.random_sample(1)
            y.append(1.0)
        else:
            x2 = w*x1-b - np.random.random_sample(1)
            y.append(-1.0)
        x.append([x1, x2])
    x = np.array(x)
    # plt.scatter(np.transpose(x)[0], np.transpose(x)[1])
    # plt.show()
    return x, y

def kernel(x, y):
    return np.dot(x,y)

def get_clip_bound(y1, y2, a1, a2, C):
    if y1 == - y2:
        L = np.max([0.0, a2-a1])
        H = np.min([C, C+a2-a1])
    elif y1 == y2:
        L = np.max([0.0, a1+a2-C])
        H = np.min([C, a1+a2])
    return L, H

def get_clip_alpha(L, H, a):
    if a>H:
        return H, True
    elif a<L:
        return L, True
    else:
        return a, False

def random_select(i,m):  # random select one number from 0,1,...,m-1 except i
    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return np.random.choice(seq)


def SMO(feature, label):
    (m, n) = feature.shape
    w = np.zeros(n)  # slope
    b = 0.0         # bias
    a = np.zeros(m)  # Lagrange variable
    C = 1e5  # initial the bound: 0<a<C

    Maxiter = 1000  # Max loop time for update a
    Epsilon = 1e-8  # convergence condition for a
    for _ in tqdm(range(100)):
        for i in range(len(a)):
            j = random_select(i, m)
            for _ in range(Maxiter):
                a1 = a[j]
                a2 = a[i]
                E1 = np.dot(w,feature[j])+b-label[j]
                E2 = np.dot(w,feature[i])+b-label[i]
                eta = kernel(feature[j], feature[j]) + kernel(feature[i], feature[i])- 2 * kernel(feature[j], feature[i])
                if eta <= 0:
                    continue
                a[i] = a2 + label[i]*(E1-E2)/eta
                # print("E2={}, E1={}, eta={}, a2={}, a2_old={}".format(E2,E1,eta,a[i+1],a2))
                L, H = get_clip_bound(label[j], label[i], a1, a2, C)
                a[i], _ = get_clip_alpha(L, H, a[i])
                a[j] = a1 + label[j] * label[i] * (a2-a[i])
                b1_new = -E1-label[j]*kernel(feature[j], feature[j])*(a[j]-a1) \
                    -label[i]*kernel(feature[i], feature[j])*(a[i]-a2)+b
                b2_new = -E2-label[i]*kernel(feature[j], feature[i])*(a[j]-a1) \
                    -label[i]*kernel(feature[i], feature[i])*(a[i]-a2)+b
                if 0<a[i]<C:
                    b=b2_new
                elif 0<a[j]<C:
                    b=b1_new
                else:
                    pass
                    # b = (b1_new+b2_new)/2.0

               
                w = np.dot(np.multiply(a, label),feature) 
                # print("lw", w, b)
                if 0<=a[i]<=C:
                    if abs(a[i]-a2) < Epsilon and abs(a[j]-a1) < Epsilon:
                        # print("break:E2={}, E1={}, eta={}, a2={}, a2_old={}".format(E2,E1,eta,a[i+1],a2))
                        break
        
    return w, b, a

def main():
    feature, label = generate_data()  # get training data
    w, b, a = SMO(feature, label)
    x = np.linspace(0,1,50)
    print(w, b)
    print(a)
    y = -(w[0]*x+b)/(w[1]+1e-6)
    plt.scatter(np.transpose(feature)[0], np.transpose(feature)[1])
    plt.scatter(np.transpose(feature)[0][np.where(a>1e-6)], np.transpose(feature)[1][np.where(a>1e-6)], c='black')
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()