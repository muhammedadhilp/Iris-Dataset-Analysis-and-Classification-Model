# Iris Dataset Analysis and Classification Model

## Introduction
This project involves analyzing the Iris dataset and building a Decision Tree model to classify the iris species. The Iris dataset contains 150 samples of iris flowers, with each sample classified into one of three species: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*. The dataset includes the following features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The goal is to use these features to build a model that predicts the species of iris flower.

---

## Dataset Description

### Loading the Data
We load the Iris dataset using pandas:

```python
import pandas as pd
df = pd.read_csv("/content/iris.csv") 
```
## Dataset Overview
A quick overview of the dataset:
```python
df.describe()
```
### Feature Descriptions:
- Sepal Length: The average sepal length is 5.843 cm, with a standard deviation of 0.828 cm.
- Sepal Width: The average sepal width is 3.054 cm, with a standard deviation of 0.433 cm.
- Petal Length: The average petal length is 3.759 cm, with a standard deviation of 1.764 cm, indicating high variability.
- Petal Width: The average petal width is 1.199 cm, with a standard deviation of 0.763 cm.
## Data Structure:
```python
df.info()
```
- The dataset contains no missing values, and the categorical species feature needs to be encoded for model training.
---
## Exploratory Data Analysis (EDA)
### Pairplot of Features
Visualize pairwise relationships between numerical features colored by species:
```python
import seaborn as sns
sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.show()
```
### Violin Plot
To compare the distribution of Sepal Length across species:

```python
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal_length', data=df, inner='quartile')
plt.show()
```
### Box Plot
Boxplot to analyze petal length distribution across species:

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_length', data=df)
plt.show()
```
### Correlation Heatmap
Compute and visualize the correlation matrix between features:
```python
corr = df.drop("species", axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()
```
Sepal Length and Petal Length are highly correlated with a correlation coefficient of 0.87.
Petal Length and Petal Width show a strong correlation of 0.92.

---
## Data Preprocessing
### Train-Test Split
We split the data into training and testing sets:
```python
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Target (species)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
```
## Feature Scaling
### Standardizing the features:

```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
```
---
## Model Training and Evaluation
### Decision Tree Classifier
We use a Decision Tree to classify the iris species:
```python
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
```
## Predictions and Accuracy
Make predictions on the test set and calculate accuracy:
```python
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```
## Classification Report
Get precision, recall, and F1-score for each species:
```python
from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred))
```
## Confusion Matrix
Plot the confusion matrix:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
---
## Results and Insights
- Accuracy: The model achieved an accuracy of 97.78% on the test set, indicating excellent performance.
- Strong Correlations: Petal length and petal width show strong correlations, which significantly influence classification.
- Species Differences: Visualizations indicate clear separations between species based on petal and sepal dimensions.
## Conclusion:
This project demonstrates the application of basic machine learning techniques, including EDA, data preprocessing, and model evaluation using the Decision Tree Classifier. The model achieved high accuracy, providing a robust prediction of iris species based on flower dimensions.

---

## Requirements
- Python 3.x
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
