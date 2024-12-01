import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("telecom_churn.csv")

# Check for num values ?
print(data.isnull().sum())

# making suitable data
y = data['Churn'].values
x_data = data.drop(['Churn'],axis=1)

# Transform values if there is different attribute from numeric
x_data = pd.get_dummies(x_data)

# normalization data
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())
x = pd.DataFrame(x,columns=x_data.columns,index=x_data.index)

# training part
from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2,random_state=42)

# importing algorithm
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(xTrain, yTrain)
print(dt.score(xTest, yTest))

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = dt.predict(xTest)

# Confusion Matrix ve Classification Report
print(confusion_matrix(yTest, y_pred))
print(classification_report(yTest, y_pred))

# ROC-AUC Skoru
y_prob = dt.predict_proba(xTest)[:, 1]
print("ROC-AUC Score:", roc_auc_score(yTest, y_prob))

# Optimize for better score
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid.fit(xTrain, yTrain)

print("Best Parameters:", grid.best_params_)
dt_optimized = grid.best_estimator_
print(dt_optimized.score(xTest, yTest))

# Visulazation importance features
importances = pd.Series(dt.feature_importances_, index=x.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importances")
plt.show()

# "Churn = 1" %rate can be low than "Churn = 0", to balance and get better score
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
xTrain_balanced, yTrain_balanced = smote.fit_resample(xTrain, yTrain)

# Visulazation of results
churn_counts = data['Churn'].value_counts()
churn_counts.plot(kind='bar')
plt.title("Churn Counts")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()


