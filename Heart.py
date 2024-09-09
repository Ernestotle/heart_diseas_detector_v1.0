# import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# import the dataset
data = pd.read_csv("framingham.csv")
data = data[data[["education", "cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]].notnull().all(1)]

features = data[['male', 'age', 'education', 'cigsPerDay', 'BPMeds',
                 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
                 'diaBP', 'BMI', 'glucose']]
label = data['TenYearCHD']

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)

# standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create the ml model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# display the results
print(lr.score(X_test, y_test))
print(lr.score(X_train, y_train))