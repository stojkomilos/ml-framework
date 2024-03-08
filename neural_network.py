import numpy as np
import matplotlib.pyplot as plt

class Dense:

    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

        if activation == 'sigmoid':
            self.func = lambda p: 1/(1+np.exp(-p))
            self.func_der = lambda p: np.exp(-p)/np.square(1+np.exp(-p))
        elif activation == 'linear':
            self.func = lambda p: p
            self.func_der = lambda p: 1
        elif activation == 'relu':
            self.func = lambda p: (p>0)*p
            self.func_der = lambda p: (p>0)
        else:
            assert False

# used mini-batch, batch, and stochastic gradient descent
class VanillaOptimizer():
    def __init__(self, lr):
        self.lr = lr

    def step(self, *, weights, biases, w_grad, b_grad):
        for lay_curr in range(len(w_grad)):
            weights[lay_curr] -= w_grad[lay_curr] * self.lr
            biases[lay_curr] -= b_grad[lay_curr] * self.lr

class AdamOptimizer():
    # TODO:
    pass

class RMSpropOptimizer():
    # TODO:
    pass


class MomentumOptimizer():
        # Andrew Ng says that beta=0.9 is a pretty good and /robust/ value for /most use cases/
        def __init__(self, lr, beta=0.9):
            self.lr = lr
            self.beta = beta

            self.weights_moving_average = None
            self.biases_moving_average = None

        def step(self, *, weights, biases, w_grad, b_grad):

            if self.weights_moving_average is None:
                self.weights_moving_average = [np.zeros_like(w) for w in w_grad]
                self.biases_moving_average = [np.zeros_like(b) for b in b_grad]

            for lay_curr in range(len(w_grad)):
                self.weights_moving_average[lay_curr] = self.beta * self.weights_moving_average[lay_curr] + (1-self.beta) * w_grad[lay_curr]
                self.biases_moving_average[lay_curr] = self.beta * self.biases_moving_average[lay_curr] + (1-self.beta) * b_grad[lay_curr]
            
            for lay_curr in range(len(w_grad)):
                weights[lay_curr] -= self.weights_moving_average[lay_curr] * self.lr
                biases[lay_curr] -= self.biases_moving_average[lay_curr] * self.lr


class Sequential:

    def __init__(self, layers, nr_inputs):
        self.layers = layers
        self.nr_layers = len(layers)
        self.nr_inputs = nr_inputs
        self.init_params()

    def init_params(self):
        self.weights = []
        self.biases = []

        for i in range(len(self.layers)):
            if i == 0:
                self.weights.append(np.random.rand(self.layers[i].units, self.nr_inputs)*np.sqrt(1/self.nr_inputs))
            else:
                self.weights.append(np.random.rand(self.layers[i].units, self.layers[i-1].units)*np.sqrt(1/self.layers[i-1].units))

            # self.biases.append(np.random.rand(self.layers[i].units, 1))
            self.biases.append(np.zeros((self.layers[i].units, 1)))

    def predict(self, X, save_cache=False):

        nr_features = X.shape[0]
        nr_samples = X.shape[1]

        a = np.array(X)

        if type(a) == np.float64:
            b = np.empty((1, 1))
            b[0] = a
            a = b

        if save_cache:
            self.a_cache = [None] * len(self.layers)
            self.z_cache = [None] * len(self.layers)

        for i in range(len(self.layers)):
            z = self.weights[i] @ a + self.biases[i]
            a = self.layers[i].func(z)

            # save for later use
            if save_cache:
                # TODO: without np.array, i dont think you need a copy
                self.a_cache[i] = np.array(a)
                self.z_cache[i] = np.array(z)
        
        return a

    def backprop(self, X, y):

        assert(self.cost == 'MSE')
        nr_layers = len(self.weights)
        nr_samples = y.shape[1]

        w_grad = [None] * nr_layers
        b_grad = [None] * nr_layers
        for curr_layer in range(nr_layers):
            w_grad[curr_layer] = np.empty(self.weights[curr_layer].shape)

        self.delta = [None] * nr_layers

        # the equations BP[1,2,3,4] are taken from https://neuralnetworksanddeeplearning.com/chap2.html

        # BP1
        if type(y) == np.float64:
            grad_C = (self.a_cache[-1] - y)
            assert(False)
        else:
            grad_C = (self.a_cache[-1] - y) / nr_samples # grad_C = (a-y)/m
        sigma_prime = self.layers[-1].func_der(self.z_cache[-1])
        self.delta[self.nr_layers-1] = grad_C * sigma_prime # calculating \delta^L

        # BP2
        for i in range(self.nr_layers-2, -1, -1):
            self.delta[i] = (self.weights[i+1].transpose() @ self.delta[i+1]) * self.layers[i].func_der(self.z_cache[i])

        # b_grad[self.nr_layers-1] = np.mean(self.delta[self.nr_layers-1], axis=1)
        # BP3
        for i in range(self.nr_layers):
            # print('i=', i, 'delta[i]=', self.delta[i])

            # Sum - svi sem poslednjeg. Mean - samo poslednji
            if i != self.nr_layers-1:
                b_grad[i] = np.sum(self.delta[i], axis=1) 
            else:
                b_grad[i] = np.mean(self.delta[i], axis=1)

            b_grad[i] = np.reshape(b_grad[i], (b_grad[i].shape[0], 1))

        # BP4
        for i in range(self.nr_layers):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    if i != 0:
                        temp = self.a_cache[i-1][k] * self.delta[i][j]
                    else:
                        # the case when a[i-1]=a[-1] but we dont want a[-1] in the python sense, but rather the input of the network
                        temp = X[k] * self.delta[i][j]

                    # sum - svi sem poslednjeg. Mean - samo poslednji
                    if i != self.nr_layers-1:
                        w_grad[i][j][k] = np.sum(temp)
                    else:
                        w_grad[i][j][k] = np.mean(temp) 
        
        return w_grad, b_grad


    def fit(self, X, y, nr_epoch, optimizer, mini_batch_size=None, use_backprop=True, pretty_output=False, check_grad=False):

        self.optimizer = optimizer

        if check_grad is True:
            assert use_backprop is True
        
        nr_samples = X.shape[1]

        # the default is to use one big batch, which is equivalent to using regular gradient descent (but only vectorised)
        if mini_batch_size is None:
            mini_batch_size = nr_samples

        nr_mini_batches = nr_samples // mini_batch_size + (nr_samples % mini_batch_size != 0)
        
        self.cost_list = np.zeros(nr_epoch)
        
        for curr_epoch in range(nr_epoch):

                if pretty_output and curr_epoch % (nr_epoch//3) == 0:
                    plt.scatter(X, self.predict(X), label='y hat')
                    plt.scatter(X, y, label='real data')
                    plt.legend()
                    plt.show()

                permutation = np.random.permutation(np.arange(0, nr_samples))
                for curr_batch in range(nr_mini_batches):
                    batch_start = curr_batch*mini_batch_size
                    batch_end = min((curr_batch+1)*mini_batch_size, nr_samples)
                    X_batch = X[:, permutation[batch_start:batch_end]]
                    y_batch = y[:, permutation[batch_start:batch_end]]

                    if use_backprop:
                        y_hat_batch = self.predict(X_batch, save_cache=True) # save_cache must be true for backprop
                        cost_batch = self.calc_cost(X_batch, y_batch)
                        w_grad, b_grad = self.backprop(X_batch, y_batch)

                        if check_grad:
                            self.check_gradient(w_grad, b_grad, X_batch, y_batch, curr_epoch)

                        # for pretty output
                        if curr_batch == 0:
                            self.cost_list[curr_epoch] = cost_batch

                            if curr_epoch % (nr_epoch//10) == 0 or (curr_epoch-1) % (nr_epoch//10) == 0:
                                print('epoch=', curr_epoch, 'cost_batch=', cost_batch)

                    else:
                        w_grad, b_grad = self.approx_gradient(X_batch, y_batch)
                    
                    # actually update the weights and biases
                    self.optimizer.step(weights=self.weights, biases=self.biases, w_grad=w_grad, b_grad=b_grad)
        
    def check_gradient(self, w_grad, b_grad, X, y, curr_epoch):
        epsilon = 10**(-7)
        margin_of_error = 10**(-5)

        w_grad_approx, b_grad_approx = self.approx_gradient(X, y, epsilon)

        for i in range(self.nr_layers):
            eucledian_norm = lambda x: np.sqrt(np.sum(np.square(x)))

            diff_w = eucledian_norm(w_grad[i] - w_grad_approx[i])
            diff_b = eucledian_norm(b_grad[i] - b_grad_approx[i])

            precision_w = diff_w / (eucledian_norm(w_grad[i]) + eucledian_norm(w_grad_approx[i]))
            precision_b = diff_b / (eucledian_norm(b_grad[i]) + eucledian_norm(b_grad_approx[i]))

            print('curr_epoch=', curr_epoch, 'cur_layer=', i)
            if precision_w > margin_of_error:
                pass
                print('bad w, precision_w=', precision_w)
                print('grad_aprrox w=', w_grad_approx[i])
                print('backprop w = ', w_grad[i])
                pass
            else:
                print("good w")
                pass
            if precision_b > margin_of_error:
                print('bad b, precision_b=', precision_b)
                print('b approx=', b_grad_approx[i])
                print('b backprop= ', b_grad[i])
                pass
            else:
                print("good b")

            print()
    
    def print_weights_and_biases(self, w, b):
        assert(len(w) == len(b))
        for lay_curr in range(len(w)):
            print(f'weights[{lay_curr}]: {w}')
            print(f'biases[{lay_curr}]: {b}')

    def set_cost_func(self, cost):
        assert cost == 'MSE'
        self.cost = cost

    def calc_cost(self, y, y_hat):
        assert(self.cost == 'MSE')

        nr_samples = y.shape[1]

        if type(y) == np.float64:
            assert False
            return np.sum(np.square(y - y_hat))/2

        if self.cost == 'MSE':
            return np.sum(np.square(y - y_hat))/2/nr_samples

    def approx_gradient(self, X, y, h=10**(-7)):

        # dont make h too small, because of numerical errors. 10**(-7) was mention by Andrew Ng as a sometimes acceptible value for gradient checking the backprop algorithm

        w_grad = []
        b_grad = []

        for curr_layer in range(len(self.weights)):
            w_grad.append(np.empty(self.weights[curr_layer].shape))
            w_grad[-1].fill(0)
            b_grad.append(np.empty(self.biases[curr_layer].shape))
            b_grad[-1].fill(0)

        # calc w gradient
        for curr_layer_w, curr_w_grad in zip(self.weights, w_grad):
            for i in range(curr_layer_w.shape[0]):
                for j in range(curr_layer_w.shape[1]):
                    old = curr_layer_w[i][j]

                    curr_layer_w[i][j] = old + h
                    a = self.calc_cost(self.predict(X), y)

                    curr_layer_w[i][j] = old - h
                    b = self.calc_cost(self.predict(X), y)

                    curr_w_grad[i][j] += (a - b)/2/h
                    curr_layer_w[i][j] = old # reset the value


        # calc b gradient
        for curr_layer_b, curr_b_grad in zip(self.biases, b_grad):
            for i in range(curr_layer_b.shape[0]):
                old = float(curr_layer_b[i]) # must be converted to float because of numpy's weirdness

                curr_layer_b[i] = old + h
                a = self.calc_cost(self.predict(X), y)

                curr_layer_b[i] = old - h
                b = self.calc_cost(self.predict(X), y)

                curr_b_grad[i] += (a - b)/2/h
                curr_layer_b[i] = old # reset the value

        nr_samples = X.shape[1]
        curr_w_grad /= nr_samples
        curr_b_grad /= nr_samples

        return w_grad, b_grad


def god_function(x):
    # return 3*x+2
    # return x * np.sin(x)
    return np.square(x)/10
    # return np.tanh(x*5)

np.random.seed(69)

# za x*sin(x)
layers = [Dense(units=7, activation='sigmoid'), Dense(units=7, activation='sigmoid'), Dense(units=1, activation='relu')]
# layers = [Dense(units=7, activation='relu'), Dense(units=7, activation='relu'), Dense(units=1, activation='linear')]
# layers = [Dense(units=4, activation='relu'), Dense(units=4, activation='relu'), Dense(units=1, activation='relu')]
# layers = [Dense(units=15, activation='sigmoid'), Dense(units=1, activation='relu')]
model = Sequential(layers, nr_inputs=1)
model.set_cost_func('MSE')

n = 64
X = np.random.uniform(low=0, high=10, size=n).T
X = X.reshape(1, n)
y = god_function(X)

# random vrednosti
optimizer = MomentumOptimizer(lr=0.5, beta=0.9)
# optimizer = VanillaOptimizer(lr=0.5)
model.fit(X, y, nr_epoch=2000, optimizer=optimizer, mini_batch_size=32)
y_hat = model.predict(X)

plt.scatter(X, y, label='real data')
plt.scatter(X, y_hat, label='y_hat')
plt.legend()
plt.show()

plt.plot(model.cost_list)
plt.show()