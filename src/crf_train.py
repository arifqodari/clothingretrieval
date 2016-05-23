import numpy as np
import argparse
import cPickle as pickle

from pystruct import learners
from pystruct.utils import SaveLogger
from pystruct import models
from sklearn import preprocessing as preproc
from sklearn import cross_validation as crosval

if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', help='training file')
    parser.add_argument('--target-file', help='target file')
    parser.add_argument('--model-filename', help='model filename')
    args = parser.parse_args()

    # load data
    X = np.load(args.data_file)
    y = np.load(args.target_file)
    y = [yy.astype(int) for yy in y]

    # preprocess data
    ef_scaler = preproc.StandardScaler()
    edge_features = X[:,2]
    ef_scaler.fit(np.vstack(edge_features))
    scaled_edge_features = []
    for edge_feature in edge_features:
        scaled_edge_features.append(ef_scaler.transform(edge_feature))
    X[:,2] = scaled_edge_features

    # save the scaler
    pickle.dump(ef_scaler, open(args.model_filename.split('.')[0] + '_scaler.pkl', 'wb'))

    # split train and test data
    test_ratio = 0.2
    X_train, X_test, y_train, y_test = crosval.train_test_split(X, y, test_size=test_ratio, random_state=0)

    # setup parameter
    n_states = np.unique(np.hstack(y)).shape[0]
    class_weight = 1. / np.bincount(np.hstack(y))
    class_weight *= float(n_states) / np.sum(class_weight)
    n_jobs = 6
    C = 0.01

    # init CRF model
    # model = models.EdgeFeatureGraphCRF(inference_method='qpbo', class_weight=class_weight, symmetric_edge_features=[0, 1], antisymmetric_edge_features=[2, 3])
    model = models.EdgeFeatureGraphCRF(inference_method='qpbo', class_weight=class_weight, symmetric_edge_features=[0, 1, 2], antisymmetric_edge_features=[3, 4])
    # model = models.EdgeFeatureGraphCRF(class_weight=class_weight, symmetric_edge_features=[0, 1], antisymmetric_edge_features=[2, 3])

    # init learner
    ssvm = learners.NSlackSSVM(model, verbose=2, n_jobs=n_jobs, C=C, logger=SaveLogger(args.model_filename, save_every=50))
    # ssvm = learners.NSlackSSVM(model, verbose=2, C=C, max_iter=100000, n_jobs=n_jobs, tol=0.0001, show_loss_every=5, logger=SaveLogger(args.model_filename, save_every=50), inactive_threshold=1e-3, inactive_window=10, batch_size=100)

    # train model
    ssvm.fit(X_train, y_train)

    # predict score on test dataset
    y_pred = ssvm.predict(X_test)
    y_pred, y_test = np.hstack(y_pred), np.hstack(y_test)
    y_pred = y_pred[y_test != 0]
    y_test = y_test[y_test != 0]

    print("Score on validation set: %f" % np.mean(y_test == y_pred))
