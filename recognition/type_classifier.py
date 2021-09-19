import numpy as np
import os
import cv2


# 这是学过一期吴恩达深度学习课程之后照猫画虎按照面向对象的方法重新做的多层神经网络
# 希望可以实现softmax
# 其实如果使用现成的框架应该会容易很多，但是我也是希望上完那个课之后，能够自己试试

class Network(object):

    def __init__(self, layer_dims: ()):
        self.layer_dims = layer_dims
        self.parameters = {}
        self.caches = []
        self.grads = {}
        self.last_cost = None

    def save_parameters(self, path='C:\\Users\\13769\\Desktop\\PROGRAMS\\TryTryTry_continue\\mats\\softmax.npy'):
        np.save(path, self.parameters)

    def load_parameters(self, path='C:\\Users\\13769\\Desktop\\PROGRAMS\\TryTryTry_continue\\mats\\softmax.npy'):
        self.parameters = np.load(path, allow_pickle=True).item()

    def initialize_parameters(self):
        length = len(self.layer_dims)
        for i in range(1, length):
            self.parameters['W' + str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) \
                                            * np.sqrt(2. / self.layer_dims[i - 1])
            self.parameters['b' + str(i)] = np.zeros((self.layer_dims[i], 1))

    def multilayer_forward(self, x_mat):
        self.caches = []
        a_mat = x_mat
        length = len(self.parameters) // 2

        for i in range(1, length):
            a_mat_prev = a_mat
            a_mat, cache = Network.linear_activation_forward(a_mat_prev, self.parameters['W' + str(i)],
                                                             self.parameters['b' + str(i)], 'relu')
            self.caches.append(cache)

        a_mat_of_l_layers, cache = Network.linear_activation_forward(a_mat, self.parameters['W' + str(length)],
                                                                     self.parameters['b' + str(length)], 'softmax')
        self.caches.append(cache)
        return a_mat_of_l_layers

    def multilayer_backward(self, AL, Y):
        self.grads = {}
        L = len(self.caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        # dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dAL = - 1 / AL
        current_cache = self.caches[-1]
        global Y_globe
        Y_globe = Y
        self.grads["dA" + str(L - 1)], self.grads["dW" + str(L)], self.grads["db" + str(L)] \
            = Network.linear_activation_backward(dAL, current_cache, 'softmax')
        for l in reversed(range(L - 1)):
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = Network.linear_activation_backward(self.grads["dA" + str(l + 1)],
                                                                                current_cache, 'relu')
            self.grads["dA" + str(l)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp

    def update_parameters(self, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l + 1)] \
                = self.parameters['W' + str(l + 1)] - learning_rate * self.grads['dW' + str(l + 1)]
            # print('dW' + str(l + 1) + ": ", self.grads['dW' + str(l + 1)])
            self.parameters["b" + str(l + 1)] \
                = self.parameters['b' + str(l + 1)] - learning_rate * self.grads['db' + str(l + 1)]
            # print('db' + str(l + 1) + ": ", self.grads['db' + str(l + 1)])

    def train(self, x_mat, y_vec, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        np.random.seed(1)
        self.last_cost = None
        for i in range(0, num_iterations):
            AL = self.multilayer_forward(x_mat)
            cost = Network.compute_cost(AL, y_vec)
            self.multilayer_backward(AL, y_vec)
            self.update_parameters(learning_rate)
            if self.last_cost is not None:
                if self.last_cost < cost:
                    print("cost value exception: value is increasing!\n loop broken")
                    break
            self.last_cost = cost
            if print_cost and (i + 1) % 10 == 0:
                print("Cost after iteration %i: %f" % (i + 1, cost))
                self.save_parameters()

    def predict(self, x_mat):
        rst = self.multilayer_forward(x_mat)
        # rst_label = np.zeros(rst.shape[1])
        rst_label = np.argmax(rst, axis=0)
        return rst_label

    def predict_pic(self, pic, size=(250, 250)):
        pic_size = pic.shape[0] * pic.shape[1]
        if pic_size < size[0] * size[1] * 0.8:
            # raise RuntimeWarning("Picture is too small!")
            return 0
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        pic = cv2.resize(pic, size)
        pic = np.reshape(pic, (pic.shape[0] * pic.shape[1], 1))
        rst = self.predict(pic)
        rst = np.squeeze(rst)
        return rst

    @staticmethod
    def linear_forward(a_mat, w_mat, b_vec):
        # print(a_mat.shape, w_mat.shape)
        z_vec = np.dot(w_mat, a_mat) + b_vec
        cache = (a_mat, w_mat, b_vec)
        return z_vec, cache

    @staticmethod
    def linear_activation_forward(a_mat_prev, w_mat, b_vec, activation='relu'):
        if activation == "sigmoid":
            z_mat, linear_cache = Network.linear_forward(a_mat_prev, w_mat, b_vec)
            a_mat, activation_cache = sigmoid(z_mat)
        elif activation == "relu":
            z_mat, linear_cache = Network.linear_forward(a_mat_prev, w_mat, b_vec)
            a_mat, activation_cache = relu(z_mat)
        elif activation == "softmax":
            z_mat, linear_cache = Network.linear_forward(a_mat_prev, w_mat, b_vec)
            a_mat, activation_cache = softmax(z_mat)
        else:
            z_mat, linear_cache = Network.linear_forward(a_mat_prev, w_mat, b_vec)
            a_mat, activation_cache = relu(z_mat)
        cache = (linear_cache, activation_cache)
        return a_mat, cache

    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        # cost = - 1 / m * np.sum(y_vec * np.log(a_mat_of_l_layers) + (1 - y_vec) * np.log(1 - a_mat_of_l_layers))
        # print(a_mat_of_l_layers.shape, y_vec.shape)
        # logg = Y * np.log(AL)
        # summ = - np.sum(Y * np.log(AL), axis=0, keepdims=True)
        cost = 1 / m * np.sum(- np.sum(Y * np.log(AL), axis=0, keepdims=True))
        cost = np.squeeze(cost)
        return cost

    @staticmethod
    def compute_m_model_cost():
        pass

    @staticmethod
    def linear_backward(dz_mat, cache):
        a_mat_prev, w_mat, b = cache
        m = a_mat_prev.shape[1]
        dw_mat = 1 / m * np.dot(dz_mat, a_mat_prev.T)
        # print("dw_mat: ", dw_mat)
        db_vec = 1 / m * np.sum(dz_mat, axis=1, keepdims=True)
        da_mat_prev = np.dot(w_mat.T, dz_mat)
        return da_mat_prev, dw_mat, db_vec

    @staticmethod
    def linear_activation_backward(da_mat, cache, activation='relu'):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dz_mat = relu_backward(da_mat, activation_cache)
            da_mat_prev, dw_mat, db_vec = Network.linear_backward(dz_mat, linear_cache)
        elif activation == "sigmoid":
            dz_mat = sigmoid_backward(da_mat, activation_cache)
            da_mat_prev, dw_mat, db_vec = Network.linear_backward(dz_mat, linear_cache)
        elif activation == "softmax":
            dz_mat = softmax_backward(da_mat, activation_cache)
            da_mat_prev, dw_mat, db_vec = Network.linear_backward(dz_mat, linear_cache)
            pass
        else:
            dz_mat = relu_backward(da_mat, activation_cache)
            da_mat_prev, dw_mat, db_vec = Network.linear_backward(dz_mat, linear_cache)
        return da_mat_prev, dw_mat, db_vec

    @staticmethod
    def load_data_sets(labels: [], path: str, size: ()):
        label_list = []
        pic_list = []
        for label in labels:
            pics = os.listdir(path + '\\' + str(label) + '\\')
            for pic in pics:
                if pic.split('.')[1] == 'png' or pic.split('.')[1] == 'jpg':
                    # print(label, labels)
                    # pic_name = pic
                    try:
                        pic = cv2.imread(path + '\\' + str(label) + '\\' + pic)
                        pic_size = pic.shape[0] * pic.shape[1]
                        if pic_size < size[0] * size[1] * 0.8:
                            continue
                        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                        pic = cv2.resize(pic, size)
                        # cv2.imshow(pic_name, pic)
                        # cv2.waitKey(1)
                        # 先按1通道来了
                        pic = np.reshape(pic, (pic.shape[0] * pic.shape[1], 1))
                    except Exception as e:
                        print(e)
                        continue

                    pic_list.append(pic)
                    label_list.append(label)

        pic_arr = np.zeros((size[0] * size[1], len(pic_list)), dtype='int32')
        for i in range(len(pic_list)):
            # print(pic_arr[:, i: i + 1])
            pic_arr[:, i: i + 1] += pic_list[i]
        pic_arr = pic_arr / 255
        '''
        print(pic_list)
        print()
        print(pic_arr)
        print('================')
        '''
        label_num = len(label_list)
        label_arr = np.zeros(label_num)
        for j in range(label_arr.shape[0]):
            label_arr[j] = int(label_list[j])
        # pic_arr = np.array(pic_list, dtype='uint8')
        # print(pic_arr)
        # pic_arr = np.reshape(pic_arr, (pic_arr.shape[0], pic_arr.shape[1])).T
        # print(pic_arr)
        # print(pic_arr, label_arr)
        return pic_arr, label_arr


# 下面的sigmoid和relu函数是按照吴恩达深度学习课程里所推到的过程直接用numpy写的，
# 可以进行整个矩阵的运算

def sigmoid(z_mat):
    a_mat = 1 / (1 + np.exp(-z_mat))
    cache = z_mat
    return a_mat, cache


def relu(z_mat):
    a_mat = np.maximum(0, z_mat)
    cache = z_mat
    return a_mat, cache


def softmax(z_mat):
    a_mat = np.exp(z_mat)
    summ = np.sum(a_mat, axis=0, keepdims=True)
    a_mat = a_mat / np.sum(a_mat, axis=0, keepdims=True)
    cache = z_mat
    return a_mat, cache


def relu_backward(da_mat, cache):
    z_mat = cache
    dz_mat = np.array(da_mat, copy=True)
    dz_mat[z_mat <= 0] = 0
    return dz_mat


def sigmoid_backward(da_mat, cache):
    z_mat = cache
    s = 1 / (1 + np.exp(-z_mat))
    dz_mat = da_mat * s * (1 - s)
    return dz_mat


def softmax_backward(da_mat, cache):
    AL = - 1 / da_mat
    dz_mat = AL - Y_globe
    return dz_mat


def one_hot(depth: int, arr):
    arr = arr.astype('int64')
    # 将整数转为一个的one hot编码
    return (np.eye(depth)[arr.reshape(-1)]).T


def train_test(path: str, save_path='D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\mats\\softmax_parameters.npy',
               labels=None, size=(250, 250), group_size=5000):
    if labels is None:
        labels = [0, 1, 2, 3]
    pic_arr, label_arr = Network.load_data_sets(labels, path, size)
    label_arr = one_hot(len(labels), label_arr)
    # print(pic_arr)
    network = Network((pic_arr.shape[0], 128, 128, 128, len(labels)))
    network.initialize_parameters()
    # network.save_parameters(save_path)
    network.train(pic_arr, label_arr, print_cost=True, num_iterations=8000, learning_rate=0.0003)
    network.save_parameters(save_path)


def predict_test(path: str, load_path='D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\mats\\softmax_parameters.npy',
                 labels=None, size=(250, 250), group_size=250):
    if labels is None:
        labels = [0, 1, 2, 3]
    pic_arr, label_arr = Network.load_data_sets(labels, path, size)
    # print(pic_arr)
    network = Network((pic_arr.shape[0], 4))
    # network.initialize_parameters()
    network.load_parameters(load_path)
    # print(pic_arr.shape, label_arr.shape)
    groups = pic_arr.shape[1] // group_size
    rest = pic_arr.shape[1] % group_size
    j = 0
    # print(network.parameters)
    predict_rst = network.predict(pic_arr)
    label_arr = label_arr.astype(np.int64)
    print(label_arr)
    print(predict_rst)
    predict_rst = np.reshape(predict_rst, label_arr.shape)
    predict_rst -= label_arr
    mistake = np.count_nonzero(predict_rst)
    mistake /= label_arr.shape[0]
    correct = 1 - mistake
    print("correct rate: %.2f" % correct)


if __name__ == '__main__':
    train_test('C:\\Users\\13769\\Desktop\\PROGRAMS\\TryTryTry_continue\\masks\\train',
               save_path='C:\\Users\\13769\\Desktop\\PROGRAMS\\TryTryTry_continue\\mats\\softmax_new.npy')
    predict_test('C:\\Users\\13769\\Desktop\\PROGRAMS\\TryTryTry_continue\\masks\\train',
                 load_path='C:\\Users\\13769\\Desktop\\PROGRAMS\\TryTryTry_continue\\mats\\softmax_new.npy')
    '''
    predict_test('D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\plates\\train',
                 load_path='D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\mats\\plate_distinguishing_test.npy')
    predict_test('D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\plates\\test',
                 load_path='D:\\MyFiles\\PROGRAMS\\python\\TryTryTry_continue\\mats\\plate_distinguishing_test.npy')
                 '''
