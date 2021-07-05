"""
File Name: Clustering.py
Objective: (Clustering)
Dependencies: 
                Pandas 
                numpy
                matplotlib
                sklearn                
"""

""" Import libraries """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sys

""" Initialise variables """
# file = "./data/wholesale_customers.csv"
file = sys.argv[-1]
PLOTS_DIR = './plots/New/'
CLUSTER_LEGEND = [ 'bo', 'rv', 'c^', 'm<', 'y>', 'ks', 'bp', 'r*', 'cD', 'mP' ]
KSet = [3, 5, 10]

""" Convert CSV to required Pandas DataFrame """
df = pd.read_csv(file)
df1 = df.drop(['Channel','Region'],axis=1)

"""Check if there are any missing values in the dataset"""
if df1.isnull().sum().sum() == 0:
    MISSING_VAL = 0
else:
    print("please handle the missing values first")
    pass
""" Question 2.1 """
if MISSING_VAL == 0:
    print("perform data analaysis")
    for col in df1.columns:
        print("{} | Mean: {} | Min: {} | Max : {} ".format(str(col),df1[col].mean(),df1[col].min(),df1[col].max()))


""" Question 2.2 """
""" Create a list of attributes from the dataframe"""
attr = [col for col in df1.columns]

""" Create a list of all pairs of attributes from the dataframe"""
attrpairs = []
for j in range(len(attr)):
    for i in range(j+1,len(attr)):
        pair = [j,i]
        attrpairs.append(pair)

""" 
Below function is used to generate required plots and calculate BC/ WC scores using K-Means Clustering 
Arguments : 
        k = number of clusters (mandatory)
        attributes = list of attributes from dataframe (optional) for 2D plot
        X, Y = attributes for 2D plots (optional)
        plot,score = optional arguments
Output :
    kmeans cluster model
    2D plots
    BC/ WC scores
"""
def KMeans_Clustering(k,attributes="",X="",Y="",plot=True,score=True):
    """ 
    Convert DataFrame to 2D array
    use Scikit Learn to identify clusters
    Usage:
        fit the model using sklearn K-means clustering
        if score is required, score == True
        if 2D plots are required, plot == true
    """
    df2array = df1.values
    kmeans = cluster.KMeans( n_clusters=k,random_state=0)
    kmeans.fit(df2array)
    
    """ Score Section """
    if score == True:
        clusterMembers = [[] for i in range(k)]
        for j in range(len(df2array)):
            clusterMembers[kmeans.labels_[j]].append(list(df2array[j]))
        
        """ Calculate Within Cluster Score """    
        WC = 0.0
        for i in range(k):
            for j in range(len(clusterMembers[i])):
                WC += (np.square(clusterMembers[i][j][0] - kmeans.cluster_centers_[i][0])+np.square(clusterMembers[i][j][1] - kmeans.cluster_centers_[i][1])+np.square(clusterMembers[i][j][2] - kmeans.cluster_centers_[i][2])+np.square(clusterMembers[i][j][3] - kmeans.cluster_centers_[i][3])+np.square(clusterMembers[i][j][4] - kmeans.cluster_centers_[i][4])+np.square(clusterMembers[i][j][5] - kmeans.cluster_centers_[i][5]))
        
        """ Calculate between Cluster Score """    
        BC = 0.0
        for i in range(k):
            for j in range(i+1,k):
                BC += np.square(kmeans.cluster_centers_[i][0] - kmeans.cluster_centers_[j][0])+np.square(kmeans.cluster_centers_[i][1] - kmeans.cluster_centers_[j][1])+np.square(kmeans.cluster_centers_[i][2] - kmeans.cluster_centers_[j][2])+np.square(kmeans.cluster_centers_[i][3] - kmeans.cluster_centers_[j][3])+np.square(kmeans.cluster_centers_[i][4] - kmeans.cluster_centers_[j][4])+np.square(kmeans.cluster_centers_[i][5] - kmeans.cluster_centers_[j][5])
        print('K={}  WC={}  BC={} BC/WC={}'.format( str(k), WC, BC, (BC/WC)))
    else:
        pass
    
    """ Plotting Section """
    if plot == True:
        plt.figure()
        for j in range(len(df2array)):
            plt.plot(df2array[j][X], df2array[j][Y], CLUSTER_LEGEND[kmeans.labels_[j]], markersize=5)
        plt.xlabel( '%r' %attributes[X]) 
        plt.ylabel( '%r' %attributes[Y])
        plt.title( 'k-means clustering %r vs %r, K= %i' %(attributes[X],attributes[Y],k))
        # plt.savefig( PLOTS_DIR + "kmeans_%r_%r_K_%i.png" %(attributes[X],attributes[Y],k))

""" Generate 2D plots for all possible combinations (15) of the attributes from the data set """
for i in range(len(attrpairs)):
    X = attrpairs[i][0]
    Y = attrpairs[i][1]
    k = KSet[0]
    KMeans_Clustering(k,attr,X,Y,plot=True,score=False)
plt.show()

""" Generate BC, WC and BC/WC Scroes for the clusters generated for Kset """
for i in range(len(KSet)):
    KMeans_Clustering(KSet[i],plot=False)

