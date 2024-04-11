import numpy as np
import pandas as pd
import pickle as pk
# import os, sys
# from matplotlib import pyplot, cm

np.seterr(all='ignore')
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(fx):
    return fx * (1 - fx)


def ReLU(Z):
    return np.maximum(0, Z)


def deriv_ReLU(Z):
    return Z > 0


def softmax(Z):
    Z = np.exp(Z)
    return (Z / np.sum(Z))


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


class NeuralNetwork:
    def __init__(self):
        self.hidden_layer0 = np.random.uniform(-0.5, 0.5, (16, 784))
        self.hidden_layer0_bias = np.random.uniform(-0.5, 0.5, (16, 1))
        self.hidden_layer1 = np.random.uniform(-0.5, 0.5, (16, 16))
        self.hidden_layer1_bias = np.random.uniform(-0.5, 0.5, (16, 1))
        self.output_layer = np.random.uniform(-0.5, 0.5, (10, 16))
        self.output_layer_bias = np.random.uniform(-0.5, 0.5, (10, 1))

    def forward_propagate(self, x_in):
        hidden_layer0_output = sigmoid((self.hidden_layer0 @ x_in) + self.hidden_layer0_bias)
        hidden_layer1_output = sigmoid((self.hidden_layer1 @ hidden_layer0_output) + self.hidden_layer1_bias)
        output_layer_output = sigmoid((self.output_layer @ hidden_layer1_output) + self.output_layer_bias)

        return hidden_layer0_output, \
               hidden_layer1_output, \
               output_layer_output

    def backward_propagate(self, learn_rate, hiddenOut0A, hiddenOut1A, outOutA, x_train, y_test):
        M = y_test.size
        one_hot_y = one_hot(y_test)
        delta_out = outOutA - one_hot_y
        self.output_layer += (1 / M) * learn_rate * (delta_out @ hiddenOut1A.T)
        self.output_layer_bias += (1 / M) * learn_rate * np.sum(delta_out)

        delta_h1 = (self.output_layer.T @ delta_out) * deriv_sigmoid(hiddenOut1A)
        self.hidden_layer1 += (1 / M) * learn_rate * (delta_h1 @ hiddenOut0A.T)
        self.hidden_layer1_bias += (1 / M) * learn_rate * np.sum(delta_h1)

        delta_h0 = (self.hidden_layer1.T @ delta_h1) * deriv_sigmoid(hiddenOut0A)
        self.hidden_layer0 += (1 / M) * learn_rate * (delta_h0 @ x_train.T) # help here
        self.hidden_layer0_bias += (1 / M) * learn_rate * np.sum(delta_h0)

    def train(self, epochs, learn_rate, x_trains, y_trains):
        for epoch in range(epochs):
            for x_train,y_train in zip(x_trains,y_trains):
                h0, h1, y_out = self.forward_propagate(x_train)
                self.backward_propagate(learn_rate, h0, h1, y_out, x_train, y_train)

                if epoch % 10 == 0:
                    prediction = get_predictions(y_out)
                    accuracy = get_accuracy(prediction, y_train)
                    print(f"Epoch: {epoch} \nAccuracy: %{100 * accuracy :0.2f}")
                    if accuracy * 100 > 90:
                        print("Reached Accuracy, Saving Model!")
                        return

    def test_input(self, index, x_test, y_test):
        x = x_test[index]
        x.shape += (1,)
        h0, h1, out = self.forward_propagate(x)
        pred = get_predictions(out)
        print(f"Prediction : {pred[0]}\nActual : {y_test[index]}")

    @staticmethod
    def save_model(obj, name='mnist_model.pickle'):
        with open(name, 'wb') as file:
            pk.dump(obj, file)
        print("Model Saved!")
    
    @staticmethod
    def load_model(name='mnist_model.pickle'):
        with open(name, 'rb') as file:
            return pk.load(file)


# def visualize(num):
#     data = pd.read_csv('train.csv')
#     data = np.array(data)
#     m, n = data.shape
#     np.random.shuffle(data)
#     data_dev = data[0: 1000].T
#     Y_dev = data_dev[0]
#     X_dev = data_dev[1:n]
#     X_dev = X_dev.T
#     visual_image = X_dev[num].reshape(28, 28)
#     fig = pyplot.figure()
#     ax1 = fig.add_subplot(122)
#     ax1.imshow(visual_image, interpolation='nearest', cmap=cm.Greys_r)
#     myNetwork = pk.load(open('mnist_model.pickle', 'rb'))
#     myNetwork.test_input(num, X_dev, Y_dev)
#     pyplot.show()


if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    data_dev = data[0: 1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev.T
    nn = NeuralNetwork()
    nn.train(10000,0.1,X_dev,Y_dev)
    nn.save_model(nn)

    #print(data)
    # nn = NeuralNetwork()
    # nn.train(10000,0.01,x_train=)
    # for _ in range(5):
    #     visualize(_)
    #     print()
