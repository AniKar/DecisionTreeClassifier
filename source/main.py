import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTreeClassifier import DecisionTreeClassifier

def main():
    # import and prepare the dataset
    data = pd.read_csv("../data/breast_cancer_data.csv")
    # drop categorcial features
    drop_list = list(set(data.columns) - set(data._get_numeric_data().columns))
    # drop some irrelevant features
    drop_list += ['Unnamed: 32','id']
    # drop some more redundant features (highly correlated with others)
    drop_list += ['perimeter_mean','radius_mean','compactness_mean',
                  'concave points_mean','radius_se','perimeter_se',
                  'radius_worst','perimeter_worst','compactness_worst',
                  'concave points_worst','compactness_se','concave points_se',
                  'texture_worst','area_worst']
    x = data.drop(drop_list, axis=1)
    y = data.diagnosis

    attributes = x.columns
    class_names = {0: 'Benign', 1: 'Malignant'}

    x = x.to_numpy()
    y = y.to_numpy()

    vf = np.vectorize(lambda x: 0 if x == 'B' else 1)
    y = vf(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # train the decision tree classifier
    # optimal tree depth found by experiments
    max_depth = 5
    dt = DecisionTreeClassifier(max_depth, attributes, class_names)
    _ = dt.fit(x_train, y_train)

    # make predictions on training data
    prd = dt.predict(x_train)
    print("Accuracy on training data", accuracy_score(y_train, prd))

    # make predictions on test data
    prd = dt.predict(x_test)
    print("Accuracy on test data", accuracy_score(y_test, prd))

    dt.plot_tree("../output/decision_tree_plot")


if __name__ == "__main__":
    main()