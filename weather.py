import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

# Download data from a url
# import requests
# path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'
# r = requests.get(path, allow_redirects=True)
# open('Weather_Data.csv', 'wb').write(r.content)

# Reading Weather_Data.csv to a data frame
df = pd.read_csv("Weather_Data.csv")

# perform one hot encoding to convert categorical variables to binary variables
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

# Droping "Date" column
df_sydney_processed.drop('Date',axis=1,inplace=True)

# Converting the data type to float
df_sydney_processed = df_sydney_processed.astype(float)

# Selecting features and dependent variable
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

# Linear Regression Part
# Spliting the data into test and train
x_train, x_test, y_train, y_test = train_test_split( features, Y, test_size=0.2, random_state=10)

LinearReg = LinearRegression()

# Traing the model on the train dateset
LinearReg.fit(x_train, y_train)

# Predicting the test dataset
predictions = LinearReg.predict(x_test)

# Calculate accuracies and errors
LinearRegression_MAE = metrics.mean_absolute_error(y_test,predictions)
LinearRegression_MSE = metrics.mean_squared_error(y_test,predictions)
LinearRegression_R2 = metrics.r2_score(y_test , predictions)

results_df = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
}

Report = pd.DataFrame(results_df)
print("Linear Regression:")
print(Report)

# KNN
KNN = KNeighborsClassifier(n_neighbors = 4).fit(x_train,y_train)

predictions = KNN.predict(x_test)
KNN_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test,predictions)
KNN_F1_Score = f1_score(y_test,predictions)

print("KNN Modeling:")
print("Accuracy score: " ,KNN_Accuracy_Score)
print("Jaccard: ",KNN_JaccardIndex)
print("F1 score: ",KNN_F1_Score)

# Decision Tree
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Tree.fit(x_train,y_train)

predictions = Tree.predict(x_test)

Tree_Accuracy_Score =  metrics.accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test,predictions)
Tree_F1_Score = f1_score(y_test,predictions)

print("Decision Tree:")
print("Accuracy score: " ,Tree_Accuracy_Score)
print("Jaccard: ",Tree_JaccardIndex)
print("F1 score: ",Tree_F1_Score)

# Logistic regression
x_train, x_test, y_train, y_test =  train_test_split( features, Y, test_size=0.2, random_state=1)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
predictions = LR.predict(x_test)
# print(predictions)

predict_proba = LR.predict_proba(x_test)
# print(predict_proba)

LR_Accuracy_Score = metrics.accuracy_score(y_test,predictions)
LR_JaccardIndex = jaccard_score(y_test,predictions)
LR_F1_Score = f1_score(y_test,predictions)
LR_Log_Loss = log_loss(y_test,predict_proba)

print("Logistic regression:")
print("Accuracy score: " ,LR_Accuracy_Score)
print("Jaccard: ",LR_JaccardIndex)
print("F1 score: ",LR_F1_Score)
print("Log loss score: ",LR_Log_Loss)
print("confusion: ", confusion_matrix(y_test,predictions))

# SVM
SVM = svm.SVC(kernel='rbf').fit(x_train, y_train) 
predictions = SVM.predict(x_test)


SVM_Accuracy_Score = accuracy_score(y_test,predictions)
SVM_JaccardIndex = jaccard_score(y_test,predictions,average="binary",pos_label=0)
SVM_F1_Score = f1_score(y_test,predictions,average='weighted')

print("Support Vector Machine:")
print("Accuracy score: " ,SVM_Accuracy_Score)
print("Jaccard: ",SVM_JaccardIndex)
print("F1 score: ",SVM_F1_Score)
print("confusion: ", confusion_matrix(y_test,predictions))

# Recreating the DataFrame since it was not saved in the previous cell
Report =  pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R2', 'Accuracy Score', 'Jaccard Index', 'F1-Score', 'Log Loss'],
    'Linear Regression': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2, np.nan, np.nan, np.nan, np.nan],
    'KNN': [np.nan, np.nan, np.nan, KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score, np.nan],
    'Tree': [np.nan, np.nan, np.nan, Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score, np.nan],
    'Logistic Regression': [np.nan, np.nan, np.nan, LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score, LR_Log_Loss],
    'SVM': [np.nan, np.nan, np.nan, SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score, np.nan]
})

print(Report)

