import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the dataset
data = pd.read_csv('path/to/dataset.csv')

# separate the features and target variable
X = data.drop('y', axis=1)
y = data['y']

# convert categorical features to numerical using label encoding
le = LabelEncoder()
X['job'] = le.fit_transform(X['job'])
X['marital'] = le.fit_transform(X['marital'])
X['educational_qual'] = le.fit_transform(X['educational_qual'])
X['call_type'] = le.fit_transform(X['call_type'])

# one-hot encode the month feature
ohe = OneHotEncoder()
month_ohe = ohe.fit_transform(X['mon'].values.reshape(-1,1))
month_ohe_df = pd.DataFrame(month_ohe.toarray(), columns=["month_"+str(int(i)) for i in range(month_ohe.shape[1])])
X = pd.concat([X, month_ohe_df], axis=1)
X = X.drop('mon', axis=1)

# scale the numerical features
scaler = StandardScaler()
X[['age', 'dur', 'num_calls']] = scaler.fit_transform(X[['age', 'dur', 'num_calls']])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# make predictions on the test set
y_pred = lr.predict(X_test)

# evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 