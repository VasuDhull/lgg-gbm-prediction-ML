import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

training_data = pd.read_csv('training_data.csv')
training_data.replace('--', None, inplace = True) # The missing values are in the form of -- so we change them to None

k = training_data.copy()
targets = pd.read_csv("training_data_targets.csv", names = ["Targets"])
k['Targets'] = targets

k = pd.get_dummies(k, columns= ['Primary_Diagnosis'], dtype= int)
k.replace('GBM', 1, inplace= True)
k.replace('LGG', 0, inplace= True)
correlation = k['Primary_Diagnosis_Glioblastoma'].corr(k['Targets'])

# Print the correlation coefficient
print("Correlation coefficient:", correlation)

print("The coorelation is so close to 1, and this leads to the model to over fit")
training_data.drop('Primary_Diagnosis',axis = 1, inplace = True)
print(training_data.isna().sum())

mode = training_data['Gender'].mode()[0]
training_data['Gender'].fillna(mode, inplace=True)

mode = training_data['Race'].mode()[0]
training_data['Race'].fillna(mode, inplace=True)

#Code to convert the age in years (Rounding off to the closest year)
k = training_data["Age_at_diagnosis"]
new_age = []
a = 0
for i in k: 
    if i == None:
        new_age.append(None)
    else:       
        l = i.split()
        if len(l) > 2:
            if float(l[2]) > (366/2):
                new_age.append(int(l[0]) + 1)
            else:
                new_age.append(int(l[0]))
        else:
            new_age.append(int(l[0]))

training_data["Age_at_diagnosis"] = new_age
print("The data set now becomes")
print(training_data)

RoundOfMean = training_data['Age_at_diagnosis'].mean().round()
training_data['Age_at_diagnosis'].fillna(RoundOfMean, inplace= True)

print("Checking for missing values")
print(training_data.isna().sum())

final_data = pd.get_dummies(training_data, dtype = int)
targets = pd.read_csv("training_data_targets.csv", names = ["Targets"])
final_data['Targets'] = targets

print("The new dataframe becomes : ")
print(final_data)
final_data.to_csv("final.csv", index = False) #To save it as a csv file if needed
scaler = MinMaxScaler() #Scaling the data of age in between 0 and 1
k = scaler.fit_transform(final_data[['Age_at_diagnosis']])
final_data[['Age_at_diagnosis']] = k

correlation_matrix = final_data.drop('Targets', axis = 1).corr()
correlation_threshold = 0.5 
highly_correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)

print(highly_correlated_features)

feature_data = final_data.drop(highly_correlated_features, axis=1)
print("### Spliting the data ###")
X = feature_data.drop("Targets", axis = 1)
y = feature_data["Targets"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print()
print("### Trainig the SVM model ###")
print()
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],  
    'gamma': ['scale', 'auto', 0.1, 1, 10], 
    'coef0': [0.0, 1.0],  
    'shrinking': [True, False], 
    'probability': [False, True], 
    'tol': [1e-3, 1e-4], 
    'max_iter': [-1, 100, 500], 
    'random_state': [42], 
}

svm = SVC()

grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=stratified_kfold, scoring='f1_macro')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

print()
print("### Training the Deision Tree ###")
print()
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'random_state': [42], 
}

dt_model = DecisionTreeClassifier()

grid_dt = GridSearchCV(dt_model, param_grid, cv=stratified_kfold, scoring='f1_macro', verbose=1)

grid_dt.fit(X_train, y_train)

print('Best Parameters:', grid_dt.best_params_)

y_pred = grid_dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
print()
print("### Training Adaptive Boosting  ###")
print()
from sklearn.ensemble import AdaBoostClassifier

param_grid = {
    'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
    'random_state': [42], 
}

adaboost = AdaBoostClassifier()

grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=stratified_kfold, scoring='f1_macro')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_adaboost = grid_search.best_estimator_
y_pred = best_adaboost.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
print()
print("### Training the LogisticRegression ###")
print()
from sklearn.linear_model import LogisticRegression
import numpy as np

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [50, 100, 200],
}

logreg = LogisticRegression()

grid_search = GridSearchCV(logreg, param_grid, cv=stratified_kfold, scoring='f1_macro', verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validated Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

y_pred = grid_search.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
print()
print("### Training the Random Forest ###")
print()

from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False],
}

random_forest_model = RandomForestClassifier(random_state=42)

grid_rf = GridSearchCV(random_forest_model, param_grid, cv=stratified_kfold, scoring='f1_macro', verbose=1)
grid_rf.fit(X_train, y_train)

print('Best Parameters:', grid_rf.best_params_)

y_pred = grid_rf.best_estimator_.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
 
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

print()
print("### Training KNN ###")
print()
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
    'leaf_size': [20, 30, 40],  # Only applicable for 'ball_tree' and 'kd_tree'
    'metric': ['minkowski', 'manhattan', 'euclidean'],  # Distance metric
}

knn_model = KNeighborsClassifier()

grid_knn = GridSearchCV(knn_model, param_grid, cv=stratified_kfold, scoring='f1_macro', verbose=1)

grid_knn.fit(X_train, y_train)

print('Best Parameters:', grid_knn.best_params_)

y_pred = grid_knn.best_estimator_.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

print("## Now we start with the test data to predict the lables ##")
test_data = pd.read_csv('test_data.csv')
k = test_data["Age_at_diagnosis"]
new_age = []
a = 0
for i in k: 
    if i == None:
        new_age.append(None)
    else:       
        l = i.split()
        if len(l) > 2:
            if float(l[2]) > (366/2):
                new_age.append(int(l[0]) + 1)
            else:
                new_age.append(int(l[0]))
        else:
            new_age.append(int(l[0]))

test_data["Age_at_diagnosis"] = new_age
print("The data set now becomes")
print(test_data)

import numpy as np
test_data.drop('Primary_Diagnosis', axis = 1, inplace = True)
final_data = pd.get_dummies(test_data, dtype = int)
scaler = MinMaxScaler() 
k = scaler.fit_transform(final_data[['Age_at_diagnosis']])
final_data[['Age_at_diagnosis']] = k
l = np.zeros(87)

feature_data = final_data.drop(highly_correlated_features, axis=1)
feature_data.insert(2, 'Race_american indian or alaska native', l)
feature_data.insert(3, 'Race_asian', l)
feature_data.insert(17, 'BCOR_MUTATED', l)
feature_data.insert(18, 'CSMD3_MUTATED', l)
print("############################################## Test Data ############################################")
print(feature_data)
print()
print()
y_pred = grid_knn.best_estimator_.predict(feature_data)
feature_data['Labels'] = y_pred
print("The final dataframe which we are left wiht is")
print()
print()
print(feature_data)
print()
print()
print("###### Labels #######")
for i in y_pred:
    print(i)