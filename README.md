# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Import and preprocess the dataset.
Split data into training and testing sets.
Train the SVM model with suitable parameters.
Test the model and evaluate accuracy.
```
## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: roopika m
RegisterNumber: 212225040348
/*
```
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories', 'Total Fat','Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target='class'
X = data[features]
y=data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm=SVC()
param_grid = {
    'C':[0.1,1,10,100],
    'kernel': ['linear','rbf'],
    'gamma': ['scale','auto'] 
}
grid_search = GridSearchCV(svm,param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model = grid_search.best_estimator_
print("Name: roopika m")
print("Register Number: 25008774")
print("Best Parameters:",grid_search.best_params_)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: roopika")
print("Register Number: 25008774")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test, y_pred))
conf_matrix= confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
 
*/
```

## Output:
```
![simple linear regression model for predicting the marks scored](sam.png)
   Calories  Total Fat  Saturated Fat  Monounsaturated Fat  \
0     149.0          0            0.0                  0.0   
1     123.0          0            0.0                  0.0   
2     150.0          0            0.0                  0.0   
3     110.0          0            0.0                  0.0   
4     143.0          0            0.0                  0.0   

   Polyunsaturated Fat  Trans Fat  Cholesterol  Sodium  Total Carbohydrate  \
0                  0.0        0.0            0     9.0                 9.8   
1                  0.0        0.0            0     5.0                 6.6   
2                  0.0        0.0            0     4.0                11.4   
3                  0.0        0.0            0     6.0                 7.0   
4                  0.0        0.0            0     7.0                13.1   

   Dietary Fiber  Sugars  Sugar Alcohol  Protein  Vitamin A  Vitamin C  \
0            0.0     0.0              0      1.3          0          0   
1            0.0     0.0              0      0.8          0          0   
2            0.0     0.0              0      1.3          0          0   
3            0.0     0.0              0      0.8          0          0   
4            0.0     0.0              0      1.0          0          0   

   Calcium  Iron  class  
0        0     0      0  
1        0     0      0  
2        0     0      0  
3        0     0      0  
4        0     0      0  
Index(['Calories', 'Total Fat', 'Saturated Fat', 'Monounsaturated Fat',
       'Polyunsaturated Fat', 'Trans Fat', 'Cholesterol', 'Sodium',
       'Total Carbohydrate', 'Dietary Fiber', 'Sugars', 'Sugar Alcohol',
       'Protein', 'Vitamin A', 'Vitamin C', 'Calcium', 'Iron', 'class'],
      dtype='object')
Name: Roopika m
Register Number: 25008774
Accuracy: 0.9794938917975567
Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.98      0.99      2003
           1       0.88      0.97      0.92       289

    accuracy                           0.98      2292
   macro avg       0.94      0.97      0.96      2292
weighted avg       0.98      0.98      0.98      2292

```
<img width="814" height="569" alt="image" src="https://github.com/user-attachments/assets/144a7a3e-846b-4260-8316-6a57ed2d29f3" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
