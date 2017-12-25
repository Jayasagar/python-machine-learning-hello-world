import numpy as np
import pandas as pd

# Load the iris data set from remote url
def loadDatasetFromUrl():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    return pd.read_csv(url, names=names)

# Entry point to stand alone program i.e. main()
def main():
    print('Main')
    dataset = loadDatasetFromUrl()
    nd_dataset= dataset.as_matrix()


# Call the main() to begin the execution
if __name__ == '__main__':
    main()