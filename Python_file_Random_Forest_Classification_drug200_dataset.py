# Random Forest Classification for drug200 Dataset

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('drug200.csv')

# Checking for null cells
data.isnull().sum()

# Replacing categorical values of Sex column by dummy variables (Because it is independent category i.e. No one is higher or lower)
# making dummies
dummy_variable_sex = pd.get_dummies(data["Sex"])
# Concatenate
data = pd.concat([data, dummy_variable_sex], axis=1)
# Drop sex column
data.drop("Sex", axis = 1, inplace=True)


# Extracting target variable
y = data.iloc[:, -3].values
# Dropping Drug column
dataset = data.drop("Drug", axis = 1, inplace=False)

# Replacing categorical values of BP and Cholesterol column by Label encoder (Because this is dependent ie higher and lower)
# Converting dataframe into arrays
dataset_array = dataset.values

# Using sklearn library's LabelEncoder class
from sklearn.preprocessing import LabelEncoder
le_BP = LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
dataset_array[:,1] = le_BP.transform(dataset_array[:,1])
dataset_array

# Using sklearn library's LabelEncoder class
from sklearn.preprocessing import LabelEncoder
le_Chole = LabelEncoder()
le_Chole.fit([ 'LOW', 'NORMAL', 'HIGH'])
dataset_array[:,2] = le_Chole.transform(dataset_array[:,2])
X = dataset_array



# Splitting the dataset into the Train set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Train set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

# Predicting a new result for a single example
print(classifier.predict(sc.transform([[23, 1, 0, 7.297999999999999, 0, 1]])))

# Predicting the entire Test set
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred, labels=['drugA','drugB','drugC','drugX','drugY'])
print(cm)
print('Accuracy is',accuracy_score(y_test, y_pred))

# Checking contents of Labels in y_test array
pd.DataFrame(y_test, columns=['Drug']).groupby('Drug').size()

# Plotting the Confusion Matrix
import itertools

# Defining th function for plotting Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
# Plotting non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['drugA','drugB','drugC','drugX','drugY'],normalize= False,  title='Confusion matrix')