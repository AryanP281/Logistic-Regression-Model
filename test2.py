#*****************************Import***********************
from LogisticRegressionModel import Logistic_Regression_Model
from feature_scaling import min_max_normalization_multivariate
import matplotlib.pyplot as plt

#*****************************Functions***********************
def get_data_classes(inputs, outputs) :
    classes = [[[], []], [[], []]]
    for i in range(0, len(outputs)) :
        if(outputs[i] == 0) :
            classes[0][0].append(inputs[i][0])
            classes[0][1].append(inputs[i][1])
        else :
            classes[1][0].append(inputs[i][0])
            classes[1][1].append(inputs[i][1])

    return classes

#*****************************Script commands***********************
inputs = []
expected_outputs = []

#Getting the training data
data_file = open("test2_data.txt", "r")
while True:
    data = data_file.readline()

    if(data != '') :
        data_sep = data.split(',')
        inputs.append([float(data_sep[0]), float(data_sep[1])])
        expected_outputs.append(int(data_sep[2]))
        continue
    
    break

data_file.close()

inputs = min_max_normalization_multivariate(inputs)

log_reg_model = Logistic_Regression_Model(2)
log_reg_model.train_with_gradient_descent(inputs[:50], expected_outputs[:50], epochs=5000, learning_rate=0.6)

accuracy = 0
for i in range(50, 100) :
    output = log_reg_model.predict(inputs[i])

    if(output >= 0.5) :
        output = 1
    else :
        output = 0

    if(output == expected_outputs[i]) :
        accuracy += 1

accuracy *= 100 / 50
print(f"Accuracy = {accuracy}%")

#Plotting the data
classes = get_data_classes(inputs, expected_outputs)
plt.plot(classes[0][0], classes[0][1], "*b", classes[1][0], classes[1][1], "+g")

#Plotting the decision boundary
x1_coords = []
x2_coords = []
for x1 in range(0,100) :
    x1 /= 100
    x1_coords.append(x1)
    x2_coords.append((-log_reg_model.parameters[0][0] - (log_reg_model.parameters[1][0] * x1)) / log_reg_model.parameters[2][0])

plt.plot(x1_coords, x2_coords, "-k")
plt.show()
