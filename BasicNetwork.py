#imports
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import make_moons
import numpy as np
np.random.seed(0)
"""
For this example I am basically going through a generated dataset and using
the functionality of a neural network (basic) to split it into two groups

Much of the inspiration from this comes from Wildml.com
"""
# first generate dataset
# sklearn make_moons makes two somewhat intersecting datasets
# like two crescent moons
X,y = make_moons( 200, noise = .20)

#naively without a neural network one could try to use a linear model here
# however the moons make that model not great. So in this case an alternate
# method should be used


#struct class for current state of the model
class NetworkModel():
    """
    This struct is a helper of NeuralNetwork and takes one as an argument
    """
    def __init__(self, nnet):
        self.W1 = np.random.randn(nnet.inDim, nnet.hiddenDim) / np.sqrt(nnet.inDim)
        self.b1 = np.zeros((1,nnet.hiddenDim))
        self.W2 = np.random.randn(nnet.hiddenDim, nnet.outDim) / np.sqrt(nnet.hiddenDim)
        self.b2 = np.zeros((1,nnet.outDim))

class NeuralNetwork():
    def __init__(self, dimIn=2, dimOut=2, dimHidden=2,eps=0.01, rlambda = 0.01):
        self.inDim = dimIn #Input dim
        self.outDim = dimOut #output dim
        self.hiddenDim  = dimHidden #hidden layer Dimension
        self.eps = eps # epsillon
        self.rLambda = rlambda # lambda for regularization str
        self.model = NetworkModel(self)

    def calc_loss(self, data_X, correct_y):
        # Calculates the current loss of the model to see how well it is doing
        # using a loss function
        # note this function assumes the activation function is tanh
        m = self.model
        N = len(data_X)
        z1 = data_X.dot(m.W1) + m.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2)
        exp_scores = np.exp(z2)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #now the loss
        correct_logprobs = -np.log(probs[range(N),y])
        data_loss = np.sum(correct_logprobs)
        # add reg term
        data_loss += self.rLambda / 2.0 * (np.sum(np.square(m.W1)) + np.sum(np.square(W2)))
        return 1.0 / N * data_loss

    def predict(self,  x):
        # using the trained model,
        # predicts output (0/1) note using tanh as activation function
        m = self.model
        z1 = x.dot(m.W1) + m.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(m.W2) + m.b2
        exp_scores = np.exp(z2)
        probs = exp_scores/np.sum(exp_score, axis=1,keepdims=True)
        return np.argmax(probs, axis=1)

    def train_model(self,X_data, y_data, n_pass = 2000):
        np.random.seed(0)
        #reinit model
        self.model = NetworkModel(self)
        m = self.model
        for i in range(n_pass):
            #forward prop
            z1 = X_data.dot(m.W1) + m.b1
            a1 = X_data.dot(z1)
            z2 = X_data.dot(m.W2) + m.b2
            exp_scores = np.exp(z2)
            probs = exp_scrores / np.sum(exp_scores, axis=1, keepdims=True)

            #back prop
            d3 = probs
            d3[range(N), y_data] -= 1
            dW2 = (m.a1.T).dot(d3)
            db2 = np.sum(d3,axis=0,keepdims=True)
            d2 = d3.dot(m.W2.T)*(1-np.power(m.a1,2))
            dW1 = np.dot(X.T,d2)
            db1 = np.sum(d2, axis=0)

            # regularization terms added
            dW2 += self.rLambda * m.W2
            dW1 += self.rLambda * m.W1
            # gradient descent
            m.W1 += -self.eps * dW1
            m.b1 += -self.eps * db1
            m.W2 += -self.eps * dW2
            m.b2 += -self.eps * db2
        return model
