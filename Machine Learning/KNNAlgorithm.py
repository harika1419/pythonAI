import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import cross_validation
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"

# Assign colum names to the dataset
field_names = ['ages', 'patient-year', 'axil-node', 'survival-data']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=field_names)

#dataset.head()


dataset.head(5)
dataset.describe()


# Split the train and test 80% and 20$
array = dataset.values
X = array[:,:3]
Y = array[:,3]
validation_size = 0.20
seed = 10
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)


#Apply KNN algorithm

X = np.array(dataset.drop(['survival-data', 'patient-year'], 1))
y = np.array(dataset['survival-data'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.20)


#knn = neighbors.KNeighborsClassifier()
#knn.fit(X_train,Y_train)

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, Y_train)


accuracy=classifier.score(X_test,Y_test)
print(accuracy)
prediction = classifier.predict(X_test)
print(prediction)

