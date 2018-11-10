##Kexin Zhai - INFO3401 - problem set 10
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression

class AnalysisData:
    def __init__(self, type1):
        self.type = type1
        self.dataset=[]
        self.variables = []

    def getData(self, filename):
        if(self.type == "csv"):        
            self.dataset = pd.read_csv(filename)
            self.variables = list(self.dataset.columns.values)
        else:
            self.dataset = open(filename).read()
        return self.dataset
    
    def get_dataVar(self, variable):
        var_col = []
        var_col = self.dataset[variable].tolist()
        return var_col                    
        
#data = AnalysisData("csv")
#data1 = data.getData("candy-data.csv")
#targetY = data.get_dataVar("chocolate")
#print(targetY)
#data2 = data.variables
#print(data2)

class LinearAnalysis:
    def __init__(self, target_Y):
        self.bestX = ""  #best X predictor for your data
        self.targetY =target_Y #index to the target dependent variable
        self.fit =-1 #how well bestX predicts your target variable
    def runSimpleAnlysis(self, data):
        dataset_1 = data.getData("candy-data.csv")
        dep_value = pd.DataFrame(dataset_1, columns = [self.targetY])
        for var in data.variables:
            if (var != 'competitorname' and var != self.targetY):
                indep_value = pd.DataFrame(dataset_1, columns = [var])
                regr = LinearRegression()
                regr.fit(indep_value, dep_value)
                coeffi_tem = regr.coef_
                intercept_tem = regr.intercept_
                predict_value = regr.predict(indep_value)
                r2Score = r2_score(dep_value, predict_value)
                if (r2Score > self.fit):
                    self.fit = r2Score
                    self.bestX = var
                    coeffi = coeffi_tem
                    intercept = intercept_tem
        print("By using linear analysis, the best x is " + self.bestX + " at " + str(self.fit))
        print("coeffient "+ str(coeffi))
        print("intercept "+ str(intercept))           
la = LinearAnalysis("chocolate")
la.runSimpleAnlysis(AnalysisData("csv"))
#get_variable = la.runSimpleAnlysis(AnalysisData("csv"))
#print(get_variable)

class LogisticAnalysis:
    def __init__(self, input_y):
        self.bestX = ""
        self.targetY = input_y
        self.fit = -1
    def runSimpleAnlysis(self, data1):
        dataset_candy = data1.getData("candy-data.csv")
        depValue = dataset_candy[self.targetY]
        for var in data1.variables:
            if (var != 'competitorname' and var != self.targetY):
                indepValue = pd.DataFrame(dataset_candy, columns = [var])
                regr = LogisticRegression()
                regr.fit(indepValue, depValue)
                co_tem = regr.coef_
                in_tem = regr.intercept_
                predictValue = regr.predict(indepValue)
                r2Score = r2_score(depValue, predictValue)
                if (r2Score > self.fit):
                    self.fit = r2Score
                    self.bestX = var
                    coeffient = co_tem
                    intercept = in_tem
        print("By using logistic analysis, the best x is " + self.bestX + " at " + str(self.fit)) 
        print("coeffient "+ str(coeffient))
        print("intercept "+ str(intercept))
    def runMultipleRegression(self, data2):
        data_m = data2.getData("candy-data.csv")
        depVal = data_m[self.targetY]
        indepVal = data_m.drop(['competitorname', self.targetY], axis = 1)
        regr = LogisticRegression()
        regr.fit(indepVal, depVal)
        predict_value = regr.predict(indepVal)
        co = []
        co.append(regr.coef_)
        print(co)
        print("intercept "+ str(regr.intercept_))
        r2Score = r2_score(depVal, predict_value)
        if (r2Score > self.fit):
            self.fit = r2Score
            
        print("By using multiple regression, the best x is " + self.bestX + " at " + str(self.fit)) 
#        print("coefficient " + regr.coef_)
#        print("intercept "+ regr.intercept_)
        
        
log = LogisticAnalysis("chocolate")
log.runSimpleAnlysis(AnalysisData("csv"))
#logM = LogisticAnalysis("chocolate")
log.runMultipleRegression(AnalysisData("csv"))
        
##1. These two functions get the same optimal variable, fruity. Linear regression fit this data better.
#
##2. Multiple Regression best fits the data since the r square score of multiple regression is around 0.76 which is larger than the r square score of simple logistic analysis, 0.43
#
##3. linear regression: y = -0.73964166x + 0.76595745
# logistic regression: p = 1/(1+e^-(-2.94124604x + 0.78465691))
#multiple regression: p = 1/(1+e^-(-2.52858047x1 - 0.19697876x2 + 0.03940308x3 - 0.16539952x4 + 0.49783674x5 - 0.47591613x6 + 0.81511886x7 - 0.59971553x8 -0.2581028x9 + 0.3224988x10 + 0.05387906x11 -1.68260553))
#
##4. (a). independent variables: candies with caramel, candies with chocolate, categorical
#dependent variable: more sugar(amount of sugar), continuous
#null hypothesis: candies with caramel and candies with chocolate has equally amount of sugar
#(b). independent variables: states (blue states v.s. red states), categorical
#dependent variable: more split ticket voters, discrete
#null hypothesis: blue states and red states has the same amount of split ticket voters.
#(c). independent variables: phones with longer battery life, other phones, categorical
#dependent variable: sell rate, continuous
#null hypothesis: phones with longer battery life sell at a same rate as other phones