import argparse
import numpy as np

from sklearn import cross_validation as crosval
from sklearn import neighbors as neigh
from sklearn import linear_model as lm
from sklearn import preprocessing as preproc
from sklearn import externals as ext


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', help='training file')
    parser.add_argument('--target-file', help='target file')
    parser.add_argument('--model-filename', help='model filename')
    args = parser.parse_args()

    # load data
    X = np.load(args.data_file)
    y = np.load(args.target_file).astype(int)

    # experiment with binary class
    # for i, yy in enumerate(y):
    #     if yy > 0:
    #         y[i] = 1

    # preprocess data
    scaler = preproc.StandardScaler()
    X = scaler.fit_transform(X)

    # split train and test data
    test_ratio = 0.2
    X_train, X_test, y_train, y_test = crosval.train_test_split(X, y, test_size=test_ratio, random_state=0)

    # learning parameter
    n_jobs = 6
    n_neighbors = 5

    # train
    clf = neigh.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', leaf_size=50, p=2, n_jobs=n_jobs)
    # clf = lm.LogisticRegression(class_weight='balanced', max_iter=100000, random_state=0, verbose=2, n_jobs=n_jobs)
    clf.fit(X_train, y_train)

    # save the model
    ext.joblib.dump((scaler, clf), args.model_filename)

    # predict score on test dataset
    print 'validation score %f' % clf.score(X_test, y_test)
