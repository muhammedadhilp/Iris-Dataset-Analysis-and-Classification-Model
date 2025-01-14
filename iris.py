# -*- coding: utf-8 -*-
"""iris.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15Dg1azQHQ39gldDGOhJ2DZsetbtLnU8D
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/iris.csv")

df.describe()
# 1.Sepal Length:
# Count: 150 (all samples have valid sepal length values).
# Mean: 5.843 cm, which is the average sepal length across all samples.
# Standard Deviation (std): 0.828 cm, indicating some variation in sepal length, but the values are relatively clustered around the mean.
# Minimum (min): 4.3 cm, the shortest sepal length recorded.
# 25th Percentile (25%): 5.1 cm, meaning 25% of the samples have a sepal length shorter than 5.1 cm.
# Median (50%): 5.8 cm, the middle value, meaning half the samples have sepal lengths shorter than 5.8 cm and half have longer.
# 75th Percentile (75%): 6.4 cm, indicating that 75% of the samples have sepal lengths shorter than 6.4 cm.
# Maximum (max): 7.9 cm, the longest sepal length recorded.
# 2. Sepal Width:
# Count: 150.
# Mean: 3.054 cm, the average sepal width across all samples.
# Standard Deviation: 0.433 cm, meaning there is less variation in sepal width compared to sepal length.
# Minimum: 2.0 cm, the narrowest sepal width recorded.
# 25th Percentile: 2.8 cm, so 25% of the flowers have a sepal width of less than 2.8 cm.
# Median: 3.0 cm, the middle value for sepal width.
# 75th Percentile: 3.3 cm, indicating that 75% of flowers have sepal widths less than 3.3 cm.
# Maximum: 4.4 cm, the widest sepal width recorded.
# 3. Petal Length:
# Count: 150.
# Mean: 3.759 cm, the average petal length across all samples.
# Standard Deviation: 1.764 cm, showing much higher variability in petal length compared to sepal dimensions.
# Minimum: 1.0 cm, the shortest petal length recorded.
# 25th Percentile: 1.6 cm, meaning that 25% of the flowers have a petal length shorter than 1.6 cm.
# Median: 4.35 cm, the middle value for petal length.
# 75th Percentile: 5.1 cm, indicating that 75% of flowers have petal lengths shorter than 5.1 cm.
# Maximum: 6.9 cm, the longest petal length recorded.
# 4. Petal Width:
# Count: 150.
# Mean: 1.199 cm, the average petal width across all samples.
# Standard Deviation: 0.763 cm, indicating a fairly wide spread of values.
# Minimum: 0.1 cm, the narrowest petal width recorded.
# 25th Percentile: 0.3 cm, meaning 25% of flowers have a petal width less than 0.3 cm.
# Median: 1.3 cm, the middle value for petal width.
# 75th Percentile: 1.8 cm, indicating that 75% of the flowers have a petal width less than 1.8 cm.
# Maximum: 2.5 cm, the widest petal width recorded.
# **Summary of Insights:
# Sepal Length and Width: The sepal length varies more than the width (higher standard deviation). Most sepal lengths are between 5.1 cm and 6.4 cm, and most sepal widths are between 2.8 cm and 3.3 cm.
# Petal Length and Width: The petal length shows significant variation (from 1.0 cm to 6.9 cm), with a mean of around 3.76 cm. The petal width also has a fair spread (from 0.1 cm to 2.5 cm), with a mean around 1.2 cm.
# Outliers: The minimum and maximum values in both petal length and width suggest the presence of distinct clusters (likely corresponding to the different species).

df.info()
# We have object dtypes which need to be encoded

df.head()

df.isna().sum()
# No null values

df.dtypes

# Pairplot of all numerical features with hue as species
sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.show()

# Violin plot to compare distributions of sepal length across species
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal_length', data=df, inner='quartile')
plt.show()

# Boxplot for comparing petal length across species
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_length', data=df)
plt.show()
# Found some outliers in setosa nad versicolor

import numpy as np

# Compute the correlation matrix
corr = df.drop("species", axis=1).corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()
# Positive correlations between sepal lenght and petal lenght and width of 0.87 and0.82 respectively
# Petal lenght and petal width has 0.92 correlations

# Facet grid for visualizing distribution of sepal length across species
g = sns.FacetGrid(df, col="species", height=5)
g.map(sns.histplot, "sepal_length")
plt.show()

# Swarm plot for sepal length by species
plt.figure(figsize=(10, 6))
sns.swarmplot(x='species', y='sepal_length', data=df)
plt.show()

# Dropping id as it is not relevant to our model
# df=df.drop("Id",axis=1)
df

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_train)
X_train=ss.transform(X_train)
X_test=ss.transform(X_test)

# model training
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train the decision tree model
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the classification report for precision, recall, f1-score
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
labels=['Iris-setosa','Iris-versicolor','Iris-virginica']
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# Accuracy: 97.78%