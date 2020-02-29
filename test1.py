#*****************************Import***********************
import random
from LogisticRegressionModel import Logistic_Regression_Model

#*****************************Script Commands***********************
inputs = []
expected_outputs = []
log_reg_model = Logistic_Regression_Model(2)

#The equation of circle is x1^2 + x2^2 = 4

#Generating the inputs and expected outputs
for i in range(0, 200) :
    x1 = random.randrange(0, 5)
    x2 = random.randrange(0, 5)

    inputs.append([x1,x2])
    if((x1**2) + (x2**2) - 4 < 0) :
        expected_outputs.append(1)
    else :
        expected_outputs.append(0)

log_reg_model.train_with_gradient_descent(inputs[:100], expected_outputs[:100], learning_rate=0.1, epochs=700)

accuracy = 0
for i in range(100,200) :
    output = log_reg_model.predict(inputs[i])

    if(output >= 0.5) :
        output = 1
    else :
        output = 0

    if(output == expected_outputs[i]) :
        accuracy += 1

print(f"Accuracy = {accuracy} %")