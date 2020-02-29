#********************************Import****************************
from LogisticRegressionModel import Logistic_Regression_Model

#********************************Script Commands****************************

#Reading the data
data_file = open("test3_data.txt", "r")
data = []
expected_outputs = []
while True :
    line = data_file.readline()
    data_sep = line.split(",")

    if(line != None) :
        data.append([float(data_sep[0]), float(data_sep[1])])
        expected_outputs.append(float(data_sep[2]))

data_file.close()

