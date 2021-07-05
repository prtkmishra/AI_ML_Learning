"""
############################################################################################################################
File Name: Classification.py
Objective: (Classification)
Dependencies: 
                Pandas 
                numpy
                sklearn       
############################################################################################################################                         
"""

""" Import libraries """
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection as model_select
import sklearn.tree as tree
import sys

"""Initialize Variables"""
# file = "./data/adult.csv"
file = sys.argv[-1]

"""Read CSV File"""
df = pd.read_csv(file)
"""Remove fnlwgt from the dataset"""
df1 = df.drop('fnlwgt',axis=1)
dfn = df1.drop('class',axis=1)
"""Create a list of attributes in the dataset"""
attr = [col for col in df1.columns]

"""Process Dataframe"""
countNumInstMissVal = dfn.isnull().any(axis=1).sum()
outdf = pd.DataFrame({  "Number of Instances": len(dfn),
                        "Number of Missing Values" : [dfn.isnull().sum().sum()],
                        "Fraction of missing values over all Attribute value" : float(dfn.isnull().sum().sum())/float(dfn.count().sum()),
                        "Number of Instances with missing values" : countNumInstMissVal,
                        "Fraction of instances with missing values over all instances": float(countNumInstMissVal)/float(len(dfn))
                        })
print("#"*100,"Q1.1","#"*100)
print(outdf.transpose(),"\n")


"""creating instances of labelencoder for every attribute"""
lenc = []
for i in range(len(attr)):
    enc = "le"+str(i)
    lenc.append(enc)

"""Replace missing values to avoid surplus labels for each missing value"""
nomDF = pd.DataFrame()
df2 = df1.replace(np.nan, "NaN")

"""Encode every Attribute"""
for i in range(len(attr)-1):
    lenc[i] = LabelEncoder()
    lenc[i].fit(df2[attr[i]])
    nomDF[attr[i]] = lenc[i].transform(df2[attr[i]])
print("#"*100,"Q1.2","#"*100)
print("(Q1.2) Set of all possible discrete values for :")
for i in range(len(attr)-1):
    """Convert to list to help remove "NaN" and inverse transform to return discrete values of every attribute"""
    x = list(lenc[i].inverse_transform(nomDF[attr[i]].unique()))
    if 'NaN' in x:
        x.remove('NaN')
    print("{} attribute: {}".format(attr[i],x))
    
print("\n")    

"""Generate a new Dataframe and remove instances with any missing values"""
nomDF3 = pd.DataFrame()
df3 = df1.dropna()

"""Refer target as class column from dataset"""
targetDF = df3['class']

"""Encode labels for all atributes except target"""
for i in range(len(attr)-1):
    nomDF3[attr[i]] = lenc[i].transform(df3[attr[i]])

"""Split the dataset into Train and Test data"""
""" Test Size has been used as 15% of the dataset as this gave the correct balance of split and better error rate"""
X_train, X_test, y_train, y_test = model_select.train_test_split( nomDF3,targetDF,test_size=0.15)
M_train = len(X_train)
M_test = len(X_test)

"""initialise the decision tree"""
clf = tree.DecisionTreeClassifier(criterion='entropy')

"""fit the tree model to the training data"""
clf.fit(X_train, y_train)

"""predict the labels for the test set"""
y_hat = clf.predict(X_test)
y_test_list = [x for x in y_test]

"""Calculate the Error rate"""
count = 0.0
for i in range( M_test ):
    if ( y_hat[i] == y_test_list[i] ):
        count += 1

error = 1 - (count/M_test)
print("#"*100,"Q1.3","#"*100)
print("(Q1.3) Error Rate : ",error,"\n")


""" Find all the instances with any missing attribute values """
missingD = df1[df1.isnull().any(axis=1)]

"""Select random samples from DF3(that contains all instances without any missing attribute values)"""
randomSample = df3.sample(n=len(missingD))

"""Merge both dataframes to create one single DataFrame Dprime"""
dprime = missingD.append(randomSample)

"""Replace all the missing values in original dataset to "missing" and fit the labelencoder"""
Mdf = df1.replace(np.nan, "missing")
for i in range(len(attr)-1):
    """Fit the new datatset into the label encoder as we have "missing" which is not covered previously"""
    lenc[i].fit(Mdf[attr[i]])

"""Create a new Dataset for Dprime1 where we replace all the missing values with "missing" """
dprime1 = dprime.replace(np.nan, "missing")

"""Create a new Dataset for Dprime2 where we replace all the missing values with mode of the attribute"""
dprime2 = dprime
for col in dprime.columns:
    if dprime[col].isnull().sum().sum() > 0:
        string = list(dprime[col].mode())
        dprime2[col] = dprime[col].replace(np.nan, string[0])
    else:
        dprime2[col] = dprime[col]

"""Create Test dataset from original data set ignoring instances from Dprime ensuring 80-20 ratio"""
xDF = Mdf.drop(dprime.index)
testDF = xDF.sample(n= int(20*((len(dprime)/80)*100)/100))

"""**Train decision tree with dprime1**"""
nomDprime1 = pd.DataFrame()
testNomDF = pd.DataFrame()
for i in range(len(attr)-1):
    nomDprime1[attr[i]] = lenc[i].transform(dprime1[attr[i]])
    testNomDF[attr[i]] = lenc[i].transform(testDF[attr[i]])
"""Refer to Original Dataset for testing"""  
targetDprime1 = dprime1['class']
X_D1_train = nomDprime1
y_D1_train = targetDprime1
X_D1_test = testNomDF
target = testDF['class']
y_D1_test = target
M_D1_train = len(X_D1_train)
M_D1_test = len(X_D1_test)
"""initialise the decision tree"""
clfDprime1 = tree.DecisionTreeClassifier(criterion= 'entropy')

"""fit the tree model to the training data"""
clfDprime1.fit(X_D1_train, y_D1_train)

"""predict the labels for the test set"""
y_D1_hat = clfDprime1.predict(X_D1_test)
y_D1_test_list = [x for x in y_D1_test]

"""Calculate the Error rate"""
countD1 = 0.0
for i in range( M_D1_test ):
    if ( y_D1_hat[i] == y_D1_test_list[i] ):
        countD1 += 1

errorD1 = 1-( countD1 / M_D1_test )
print("#"*100,"Q1.4","#"*100)
print('Error rate for Dprime1 : ',errorD1,"\n")

"""**Train decision tree with dprime2 and test with original data**"""
nomDprime2 = pd.DataFrame()
for i in range(len(attr)-1):
    nomDprime2[attr[i]] = lenc[i].transform(dprime2[attr[i]])
    
targetDprime2 = dprime2['class']
X_D2_train = nomDprime2
y_D2_train = targetDprime2
X_D2_test = testNomDF
target = testDF['class']
y_D2_test = target
M_D2_train = len( X_D2_train )
M_D2_test = len( X_D2_test )
"""initialise the decision tree"""
clfDprime2 = tree.DecisionTreeClassifier(criterion='entropy',ccp_alpha=0.0)

"""fit the tree model to the training data"""
clfDprime2.fit( X_D2_train, y_D2_train )

"""predict the labels for the test set"""
y_D2_hat = clfDprime2.predict(X_D2_test)
y_D2_test_list = [x for x in y_D2_test]

"""Calculate the Error rate"""
countD2 = 0.0
for i in range( M_D1_test ):
    if ( y_D2_hat[i] == y_D2_test_list[i] ):
        countD2 += 1
errorD2 = 1 - ( countD2 / M_D2_test )
print('Error rate for Dprime2 : ',errorD2,"\n")