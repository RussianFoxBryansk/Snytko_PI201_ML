import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
        o1 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
        return o1

    def train(self, data, all_y_trues, learn_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
                h2 = sigmoid(sum_h2)

                sum_h3 = self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3
                h3 = sigmoid(sum_h3)

                sum_o1 = self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4
                y_pred = sigmoid(sum_o1)

                d_L_d_ypred = -2 * (y_true - y_pred)
                self._update_weights(x, d_L_d_ypred, y_pred, h1, h2, h3, learn_rate)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

    def _update_weights(self, x, d_L_d_ypred, y_pred, h1, h2, h3, learn_rate):
        d_ypred_d_w10 = h1 * deriv_sigmoid(y_pred)
        d_ypred_d_w11 = h2 * deriv_sigmoid(y_pred)
        d_ypred_d_w12 = h3 * deriv_sigmoid(y_pred)
        d_ypred_d_b4 = deriv_sigmoid(y_pred)

        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
        d_h1 = deriv_sigmoid(sum_h1)

        sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
        d_h2 = deriv_sigmoid(sum_h2)

        sum_h3 = self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3
        d_h3 = deriv_sigmoid(sum_h3)

        self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_w10
        self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_w11
        self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_w12
        self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_b4

        self.w1 -= learn_rate * d_L_d_ypred * self.w10 * d_h1 * x[0]
        self.w2 -= learn_rate * d_L_d_ypred * self.w10 * d_h1 * x[1]
        self.w3 -= learn_rate * d_L_d_ypred * self.w10 * d_h1 * x[2]
        self.b1 -= learn_rate * d_L_d_ypred * self.w10 * d_h1

        self.w4 -= learn_rate * d_L_d_ypred * self.w11 * d_h2 * x[0]
        self.w5 -= learn_rate * d_L_d_ypred * self.w11 * d_h2 * x[1]
        self.w6 -= learn_rate * d_L_d_ypred * self.w11 * d_h2 * x[2]
        self.b2 -= learn_rate * d_L_d_ypred * self.w11 * d_h2

        self.w7 -= learn_rate * d_L_d_ypred * self.w12 * d_h3 * x[0]
        self.w8 -= learn_rate * d_L_d_ypred * self.w12 * d_h3 * x[1]
        self.w9 -= learn_rate * d_L_d_ypred * self.w12 * d_h3 * x[2]
        self.b3 -= learn_rate * d_L_d_ypred * self.w12 * d_h3

    def save_weights(self, filename):
        weights_and_biases = np.array([
            self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7,
            self.w8, self.w9, self.w10, self.w11, self.w12,
            self.b1, self.b2, self.b3, self.b4
        ])
        np.savetxt(filename, weights_and_biases)

    def load_weights(self, filename):
        weights_and_biases = np.loadtxt(filename)
        (self.w1, self.w2, self.w3, self.w4, self.w5, self.w6,
         self.w7, self.w8, self.w9, self.w10, self.w11, self.w12,
         self.b1, self.b2, self.b3, self.b4) = weights_and_biases
