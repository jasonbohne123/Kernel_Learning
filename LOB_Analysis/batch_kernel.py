from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import sys

path='/home/jbohn/jupyter/personal/Kernel_Learning/'
sys.path.append(path)
from Kernels.kernel import Kernel
from Kernels.helpers import compose_kernels


def train_svm_batch(batched_data, batch_estimates, kernel_type, order):
    """ Trains SVM across batch of data 
    """
    batch_index = {}
    batched_svm = {}

    counter = 0

    # initialize kernel types
    if kernel_type == 'linear':
        kernel_list = [Kernel(kernel_type=kernel_type)
                       for i in range(1, order+1)]
    elif kernel_type == 'polynomial':

        kernel_list = [Kernel(kernel_type, i) for i in range(1, order+1)]
    elif kernel_type == 'gaussian':
        kernel_list = [Kernel(kernel_type, bandwidth=i)
                       for i in np.linspace(0.1, 1, order)]  # gamma hyperparam
    else:
        print("Not Valid Kernel Type")
        return

    for batch, data in batched_data.items():

        # get data
        X, y = data["features"], data["outcomes"]

        # compose kernel
        kernel = compose_kernels(X, kernel_list, batch_estimates[batch])
        
        # fit svm
        clf = svm.SVC(kernel='precomputed')
        clf.fit(kernel, y)

        # save kernel
        batched_svm[batch] = clf
        batch_index[counter] = batch
        counter+=1

    return batched_svm, batch_index


def predict_svm_batch(batched_kernels, batch_index, batched_data,batch_estimates ,kernel_type, order):
    """ Predicts SVM across batch of data; saving accuracy, recall, precision
    """
    batched_predictions = {}

    # initialize kernel types
    if kernel_type == 'linear':
        kernel_list = [Kernel(kernel_type=kernel_type)
                       for i in range(1, order+1)]
    elif kernel_type == 'polynomial':

        kernel_list = [Kernel(kernel_type, i) for i in range(1, order+1)]
    elif kernel_type == 'gaussian':
        kernel_list = [Kernel(kernel_type, bandwidth=i)
                       for i in np.linspace(0.1, 1, order)]  # gamma hyperparam
    else:
        print("Not Valid Kernel Type")
        return

    counter = 0
    for batch, trained_svm in batched_kernels.items():

        # evaluate kernel on subsequent batch
        next_batch = batch_index[counter+1]

        # get data
        X = batched_data[next_batch]["features"].values
        true = batched_data[next_batch]["outcomes"].values

        #predict svm; need to precompute test kernel
        oos_kernel=compose_kernels(X, kernel_list, batch_estimates[batch])
        y_pred = trained_svm.predict(oos_kernel )

    
        # compute accuracy
        accuracy = accuracy_score(true, y_pred)

        # compute weighted recall
        recall = recall_score(true, y_pred, average='weighted')


        # evaluation dictionary
        evaluation_dict = {"accuracy": accuracy,"recall": recall}

        # save predictions
        batched_predictions[batch] = evaluation_dict

    return batched_predictions
