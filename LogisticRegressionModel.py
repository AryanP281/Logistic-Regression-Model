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

        #Converting the inputs to a matrix
        X = self.get_input_matrix(training_inputs)

        #Converting the expected outputs to a matrix
        y = np.array(expected_outputs)
        y.shape = (1, len(expected_outputs))

        #Training for the epochs
        for epoch in range(0, epochs) :
            derivatives = np.zeros(self.parameters.shape) #The derivatives for the parameters

            #Iterating through the dataset
            for inp in range(0, X.shape[0]) :
                x = X[inp] #The input set
                z = np.dot(self.parameters.T, x.T) #Calculating theta' * x'
                h = self.sigmoid(z) #Getting sigmoid of z
                error = h - y[0][inp] #Calculating the error in the model's prediction

                #Updating the derivatives
                for der in range(0,derivatives.shape[0]) :
                    derivatives[der][0] += error[0] * x[der]

                #Updating the cost function value
                self.cost_func_value += -(y[0][inp] * log(h)) - ((1.0 - y[0][inp]) * log(1.0 - h))

            #Calculating the final derivatives
            derivatives = derivatives / X.shape[0]

            #Calculating the final value of the cost function
            self.cost_func_value /= X.shape[0]

            #Checking if regularization is to be used
            if(use_regularization) :
                param_sq_sum = 0 #The sum of the parameters squared (required for calculating cost function)
                for param in range(1, self.parameters.shape[0]) :
                    derivatives[param][0] += lmbda * self.parameters[param][0] / X.shape[0]
                    param_sq_sum += self.parameters[param][0] ** 2

                #Calculating the final value of the cost function
                self.cost_func_value += lmbda * param_sq_sum / (2 * X.shape[0])

            #Updating the parameters
            self.update_parameters(derivatives, learning_rate)

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