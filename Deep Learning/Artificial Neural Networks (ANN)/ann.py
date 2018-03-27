import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
#
#classifier = Sequential()

# input layer and 1st hidden layer
#classifier.add(Dense(6, kernel_initializer="uniform", activation='relu', input_dim=11))
#
## 2nd hidden layer
#classifier.add(Dense(6, kernel_initializer="uniform", activation='relu'))
#
## output layer
#classifier.add(Dense(1, kernel_initializer="uniform", activation='sigmoid'))
#
#classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

#Y_pred = classifier.predict(X_test)
#Y_pred = (Y_pred > 0.5)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_test, Y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer="uniform", activation='relu', input_dim=11))
#    classifier.add(Dropout(rate=0.1, ))
#    classifier.add(Dense(6, kernel_initializer="uniform", activation='relu'))
#    classifier.add(Dropout(rate=0.1, ))
    classifier.add(Dense(1, kernel_initializer="uniform", activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 300)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = 1)

mean = accuracies.mean()
variance = accuracies.std()

# tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer="uniform", activation='relu', input_dim=11))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(6, kernel_initializer="uniform", activation='relu'))
#    classifier.add(Dropout(rate=0.1, ))
    classifier.add(Dense(1, kernel_initializer="uniform", activation='sigmoid'))
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size=32, epochs=300)
accuracies_edit = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = 1)

#parameters = {'batch_size': [25, 32],
#              'epochs': [100, 500],
#              'optimizer': ['adam', 'rmsprop']}
#
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv=10)
#
#grid_search = grid_search.fit(X_train, Y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_
