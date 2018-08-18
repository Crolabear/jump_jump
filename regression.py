import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


a=pd.read_csv('train2.csv')
columns1 = ['perX',	'perY',	'topLeftX',	'topLeftY',	'botRightX','botRightY','matchedIndex']
labeledOutcome = a['timePress']
m = len(labeledOutcome)
predictors =  pd.DataFrame(a, columns=columns1)
# predictors1 =  pd.DataFrame(a, columns=columns1,np.ones(m))
labeledOutcome = a['timePress']
model = LinearRegression()
model.fit(predictors, labeledOutcome)

i = 0
meanSq = []
for i in range(60):
    oneOff = a[columns1][i:i+1]
    target = a['timePress'][i]
    # model.fit(predictors, target)
    yfit = model.predict(oneOff)
    meanSq.append((yfit[0]-target)* (yfit[0]-target))
# print (np.sqrt(np.mean(meanSq)))
#
# personX = a['perX'].values
# personY = a['perY'].values
# topLeftX = a['topLeftX'].values
# topLeftY = a['topLeftY'].values
# botRightX = a['botRightX'].values
# botRightY = a['botRightY'].values
# matchedIndex = a['matchedIndex'].values
# m = len(matchedIndex)
# bias = np.ones(m)
# X = np.array([bias, personX, personY,topLeftX,topLeftY,botRightX,botRightY,matchedIndex]).T
#
# B = np.array([0, 0, 0])
# Y = np.array(write)
