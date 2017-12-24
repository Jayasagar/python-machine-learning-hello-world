# Load libraries

import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# read training set data from the CSV format
def getTrainingDataset():
    trainingDataset = []
    inputAttributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv('iris.data.txt', names=inputAttributes)

    # print('my_dataframe.columns.values.tolist()', list(df))
    # numpyMatrix = dataset.as_matrix()
    # #print('at index dataframe:', df[0])
    # print('NumPy Dataset Array using Pandas', numpyMatrix)
    # return numpyMatrix
    return dataset

# Load the iris data set sklearn
def loadPreBuiltIrisDataset():
   return datasets.load_iris()

# Load the iris data set from remote url
def loadDatasetFromUrl():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    return pd.read_csv(url, names=names)

def buildModel(X_train, y_train, algorithm, model):
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold,
                                                 scoring='accuracy')
    print('cv_results:', cv_results)
    result = "Cross Verification Result -> %s:   %f (%f)" % (algorithm, cv_results.mean(), cv_results.std())
    print(result)
    return cv_results

def predict(X_train, X_validation, y_train, y_validation, algorithm, model):
    # Prediction Report
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    print('Accuracy score:', algorithm, accuracy_score(y_validation, predictions))
    print('Confusion Matrix', confusion_matrix(y_validation, predictions))
    print('Classification report \n', algorithm, classification_report(y_validation, predictions))

# Entry point to stand alone program i.e. main()
def main():
    print('Main')

    #dataset = getTrainingDataset()
    dataset = loadPreBuiltIrisDataset()
    ## Fields: {data, target, target_names,}
    #print('dataset', dataset)
    X = dataset.data
    Y = dataset.target
    iris_dataframe = pd.DataFrame(X, columns=dataset.feature_names)

    # Shape
    shape = iris_dataframe.shape
    print('Shape:', shape)
    print('Number of Dimensions:', len(shape)) # 2

    # Statistical Summary(count, mean, min,max, etc…)
    print(iris_dataframe.describe())

    # Univariate plots
    # subplots=True: Make separate subplots for each column
    # layout=(2, 2): (rows, columns) for the layout of subplots
    iris_dataframe.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    #plt.show()

    # histograms
    #iris_dataframe.hist()
    #plt.show()

    # X = dataset.values[:, 0:4]  # Slice Input attribute value from 0 to 3 index
    # y = dataset.values[:, 4]  # Slice Output attribute values

    # Split both input/output attribute arrays into random train and test subsets
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20)

    #iris_dataframe = pd.DataFrame(X, columns=dataset.feature_names)

    # Create a scatter matrix from the dataframe, color by y_train
    pd.plotting.scatter_matrix(iris_dataframe, c=Y, figsize=(10, 10), marker='.',
                                     hist_kwds={'bins': 20}, s=60, alpha=.8)


    algorithm_dict = {
        'CART': DecisionTreeClassifier(),
        'NB': GaussianNB(),
        'SVC': SVC(),
        'K-N': KNeighborsClassifier(),
        'LR': LogisticRegression(),
        'LDA': LinearDiscriminantAnalysis()
    }

    results = []

    # Iterate through all the Algorithms and Cross Verify with test dataset
    for key, value in algorithm_dict.items():

        # Build the model
        result = buildModel(X_train, Y_train, key, value)
        results.append(result)

        # Predict the model on test data
        predict(X_train, X_validation, Y_train, Y_validation, key, value)


    # Compare Cross Verification Results
    fig = plt.figure()
    fig.suptitle('Models Comparison')
    # axis = fig.add_subplot(1, 1, 1)
    plt.boxplot(results, labels=algorithm_dict.keys(), showmeans=True, meanline=True)
    # axis.set_xticklabels(algorithm_dict.keys())
    plt.show()


# Call the main() to begin the execution
if __name__ == '__main__':
    main()