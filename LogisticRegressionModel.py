#*****************************Import***********************
import numpy as np
from math import e, log

#*****************************Classes***********************
class Logistic_Regression_Model(object) :

    x0 = 1 #The value of the bias

    def __init__(self, features_count) :
        """
        features_count = number of features considered
        """

        #Initializing the features list
        self.parameters = np.zeros(features_count + 1) #A NumPy array containing the model's parameters
        self.parameters.shape = (features_count + 1, 1) #Saving the parameters as a column vector i.e m*1 matrix

        #Initializing the model's attributes
        self.cost_func_value = 0.0 #The value of the cost function J(theta) for the last training epoch

    def train_with_gradient_descent(self, training_inputs, expected_outputs, epochs, learning_rate = 0.001, use_regularization=False, lmbda = 0) :
        """Trains the model using gradient descent"""

        X = np.concatenate((np.ones((len(training_inputs), 1)), training_inputs), axis=1) #Converting the training inputs to NumPy array

        #Converting the expected outputs to a matrix
        y = np.array(expected_outputs)
        y.shape = (len(expected_outputs), 1)

        #Training for the epochs
        for epoch in range(0, epochs) :
            #Forward Propogation
            H, E, self.cost_func_value = self.forwardpropogate(X, y, self.parameters, lmbda)

            #Getting the gradients
            derivatives = self.get_gradients(self.parameters, X, y, E, lmbda)

            #Updating the parameters
            self.update_parameters(derivatives, learning_rate)

    def forwardpropogate(self, X, y, parameters, lmbda) :
        """Forward propogates and returns the outputs and cost function value for the given inputs.
        X = inputs
        y = expected outputs
        parameters = parameters to be used
        lmbda = Regularization Parameter
        returns (Outputs, Errors, Cost)"""

        #The model output using the given parameters
        Z = np.dot(X, parameters)
        H = self.sigmoid(Z)

        #Calculating the cost function value
        E = H - y
        J1 = np.concatenate((-y, -(1 - y)), axis=1)
        J2 = np.concatenate((np.log(H), np.log(1 - H)), axis=1)
        j = np.sum(J1 * J2) / y.shape[0]
        j += (lmbda / (2 * y.shape[0])) * (np.sum(y[1:-1]**2)) 

        #Returning the outputs and cost
        return (H, E, j)

    def get_gradients(self, params, X, y, E, lmbda) :
        """Calculates and returns the gradients for the parameters
        params = the parameters whose gradients are required,
        X = inputs,
        y = expected outputs,
        E = errors in the model's outputs i.e (h - y),
        lmbda = Regularization Parameter
        return grads = a Numpy array containing the calculated gradients"""

        grads = np.zeros(params.shape) #The gradients

        #Calculating the gradients
        grads = np.dot(X.T, E) / y.shape[0]  
        grads[1:] += params[1:] * (lmbda / y.shape[0])

        #Returning the calculated gradients
        return grads

    def get_input_matrix(self, inputs) :
        """Converts the input list to a NumPy matrix containing the inputs as well as bias"""

        #Creating a matrix containing the bias and inputs
        X = np.array([self.x0] + inputs[0])

        #Adding rows to the matrix
        for inp in range(1, len(inputs)) :
            X = np.vstack((X, [self.x0] + inputs[inp]))

        #Setting the shape of the matrix
        X.shape = (len(inputs), self.parameters.shape[0])

        return X

    def sigmoid(self, z) :
        """Returns sigmoid of z"""

        return (1.0 / (1.0 + (e ** -z)))

    def update_parameters(self, derivatives, learning_rate) :
        """Updates the parameters of the model"""

        for param in range(0, self.parameters.shape[0]) :
            self.parameters[param][0] -= learning_rate * derivatives[param][0]

    def predict(self, inputs) :
        """Returns the models prediction for the given inputs"""

        #Initializing the input vector
        x = np.array([self.x0] + inputs)
        x.shape = (len(inputs) + 1, 1)

        #Calculating the value of z i.e theta' * x
        z = np.dot(self.parameters.T, x)

        #Calculating the sigmoid value
        h = self.sigmoid(z)

        return h