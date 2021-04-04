# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:33:04 2020

@author: karthik.ragunathan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import seaborn as sns

## Step 0: Business understanding, Data cleaning, Preprocessing
## Step 1: Exploratory analysis
## Step 2: Scale the data if needed
## Step 3: Generate clusters; KMeans
  # Generally business teams will have an expectation on the number of clusters
  # Elbow curve method can be used to check for optimal number of clusters for a given data
  
## Step 4: Cluster Profiling
  # Reference: https://en.wikipedia.org/wiki/Claritas_Prizm

####### Importance of Scaling in Multivariate analysis ########################
#Given is the age, income and height of 5 individuals (P1,P2,P3,P4 and P5)
age = [32,45,28,60,55]
income = [25000,55000,35000,18000,42000]
height = [165, 161, 145, 180, 170]
#Who is similar to P1 based on age, income and height?

#d = sqrt((x1-x2)^2 + (y1 - y2)^2 + (z1 - z2)^2) # Euclidean distance
plt.scatter(age,income)

p1p2 = np.sqrt((32 - 45)**2 + (25000 - 55000)**2) #30000
p1p3 = np.sqrt((32 - 28)**2 + (25000 - 35000)**2) #10000
p1p4 = np.sqrt((32 - 60)**2 + (25000 - 18000)**2) #7000
p1p5 = np.sqrt((32 - 55)**2 + (25000 - 42000)**2) #17000
# p1 p4 is appeairing to be closer because income has biased because of it's scale


p1p2 = np.sqrt((32 - 45)**2 + (25 - 55)**2) #32
p1p3 = np.sqrt((32 - 28)**2 + (25 - 35)**2) #10.7
p1p4 = np.sqrt((32 - 60)**2 + (25 - 18)**2) #28.8
p1p5 = np.sqrt((32 - 55)**2 + (25 - 42)**2) #28.6
# above approach of scaling is not methodical

## Standardization
# (x - mu)/sigma
# ALl the variables will become 0 mean, 1 standard deviation/variance
age_scaled = scale(age)
np.mean(age_scaled)
np.std(age_scaled)
inc_scaled = scale(income)
np.mean(inc_scaled)
np.std(inc_scaled)
ht_scaled = scale(height)
np.mean(ht_scaled)
np.std(ht_scaled)
p1p2 = np.sqrt((age_scaled[0] - age_scaled[1])**2 + (inc_scaled[0] - inc_scaled[1])**2) #2.54
p1p3 = np.sqrt((age_scaled[0] - age_scaled[2])**2 + (inc_scaled[0] - inc_scaled[2])**2) #0.836
p1p4 = np.sqrt((age_scaled[0] - age_scaled[3])**2 + (inc_scaled[0] - inc_scaled[3])**2) #2.3
p1p5 = np.sqrt((age_scaled[0] - age_scaled[4])**2 + (inc_scaled[0] - inc_scaled[4])**2) #2.26

###############
irisdata = pd.read_csv("data/iris.csv")
newiris = irisdata.iloc[:,:4]

## Step 1: Exploratory Analysis
sns.pairplot(newiris)
# Petal Length and Petal Width is giving good separability
# Sepal width has weak separability

iris_summary = newiris.describe()
newiris.std()/newiris.mean()
# P.L and P.W have a high variance

newiris.plot.scatter("Petal.Length","Petal.Width")

## Step 2
# Scaling not needed as all variables are measurement in cms

## Step 3

iris_with_label2 = newiris.copy()
iris_kmeans2 = KMeans(n_clusters=2, random_state = 1234).fit(newiris)
iris_kmeans2.labels_
iris_with_label2["Cluster"] = iris_kmeans2.labels_

sns.scatterplot("Sepal.Length","Sepal.Width", data = newiris)
sns.scatterplot("Sepal.Length","Sepal.Width", data = iris_with_label2, hue = "Cluster")
sns.scatterplot("Petal.Length","Petal.Width", data = iris_with_label2, hue = "Cluster")
sns.scatterplot("Petal.Length","Sepal.Length", data = iris_with_label2, hue = "Cluster")
sns.pairplot(iris_with_label2, hue = "Cluster")

iris_with_label3 = newiris.copy()
iris_kmeans3 = KMeans(n_clusters=3, random_state = 1234).fit(newiris)
iris_with_label3["Cluster"] = iris_kmeans3.labels_

sns.scatterplot("Sepal.Length","Sepal.Width", data = iris_with_label3, hue = "Cluster")
sns.scatterplot("Petal.Length","Petal.Width", data = iris_with_label3, hue = "Cluster")
sns.scatterplot("Petal.Length","Sepal.Length", data = iris_with_label3, hue = "Cluster")
sns.pairplot(iris_with_label3, hue = "Cluster")

## Elbow Curve Analysis

# Inertia is Within cluster distance
iris_kmeans2.inertia_ #152
iris_kmeans3.inertia_ #79

# Trying till 5 clusters
iris_withinclust_dist = pd.Series(0.0, index = range(1,6))
for k in range(1,6):
    iris_anyk = KMeans(n_clusters=k, random_state = 1234).fit(newiris)
    iris_withinclust_dist[k] = iris_anyk.inertia_

iris_withinclust_dist.plot.line()
# Elbow point at K = 2.. Within cluster distance saturates after k = 3
# This data can be optimally clustered into 2 segments or at the maximum 3 segments. 

## Step 4

iris_clust_profile = iris_with_label2.groupby("Cluster").agg(np.mean)

for i in iris_clust_profile.columns:
    plt.figure()
    iris_clust_profile[i].plot.bar()
    plt.title(i)

sns.pairplot(iris_with_label2, hue = "Cluster")

# Cluster 0: Small flowers with Less P.L, P.W and S.L
# Cluster 1: Large Flowers with high P.L, P.W and S.L

# There is not much difference based on S.W

################### Wine #####################################

# Link for Data: https://archive.ics.uci.edu/ml/datasets/Wine

# You have data for 178 wine with 13 attributes
# Cluster 178 wine into segments
# You can find the optimal number of clusters

# Note:
# It is a csv file. just that the extension is .data
# There will be total 14 columns. Ignore the first column. 
# Data doesn't have column header. It has to be added separately
# 13 attributes are of different scale. So they have to be brought to a comparable scale
    # inbuilt scale function can scale all columns in a dataframe. output will be a numpy matrix

## Step 0
winedata = pd.read_csv("data/wine.data", header = None)
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280_OD315",
                       "Proline"]
newwine = winedata.iloc[:,1:]

## Step 1
wine_summary = newwine.describe()
sns.pairplot(newwine)

## Step 2
wine_scaled = pd.DataFrame(scale(newwine), columns = newwine.columns)
wine_scaled_summary = wine_scaled.describe()
# all variables became 0 mean, unit variance

# Scaling doesn't impact any relationship between variables
# If you visually look at the scatter plot, there won't be any difference between raw vs scaled
newwine.plot.scatter("OD280_OD315","Proline") # raw
wine_scaled.plot.scatter("OD280_OD315","Proline") # scaled

newwine.plot.scatter("Alcohol","Hue") # raw
wine_scaled.plot.scatter("Alcohol","Hue") # scaled

## Step 3

wine_withinclust_dist = pd.Series(0.0, index = range(1,10))
for k in range(1,10):
    wine_anyk = KMeans(n_clusters=k, random_state = 1234).fit(wine_scaled)
    wine_withinclust_dist[k] = wine_anyk.inertia_

wine_withinclust_dist.plot.line()
# Elbow at K=3

wineclust3 = KMeans(n_clusters=2, random_state = 1234).fit(wine_scaled)
wine_with_label = newwine.copy()
wine_with_label["Cluster"] = wineclust3.labels_

## Step 4
wine_clust_profile = wine_with_label.groupby("Cluster").agg(np.mean)

for i in wine_clust_profile.columns:
    plt.figure()
    wine_clust_profile[i].plot.bar()
    plt.title(i)

for i in wine_clust_profile.columns:
    plt.figure()
    wine_with_label.boxplot(column = i, by = "Cluster")
    plt.title(i)

sns.pairplot(wine_with_label, hue = "Cluster")

# Cluster 0: High Proline, Flavanoids, Phenols, OD280; Less alcalanity of ash
# Cluster 1: High Color Intensity, Non flavanoid phenols, Malic acid; Less OD280, Hue, Pronathocyanins, Flavanoids
# Cluster 2: Less Color intensity, Alcohol
