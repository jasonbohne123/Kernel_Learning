from sklearn import svm
from sklearn.metrics import accuracy_score,recall_score,precision_score

def train_svm_batch(batched_data):
    """ Trains SVM across batch of data 
    """
    batch_index={}
    batched_svm={}

    counter=0
    for batch,data in batched_data.items():
        # get data
        X,y=batched_data["features"],batched_data["outcomes"]
        # fit svm
        clf=svm.SVC(kernel='precomputed')
        clf.fit(X,y)
        # save kernel
        batched_svm[batch]=clf
        batch_index[counter]=batch

    return batched_svm,batch_index

def predict_svm_batch(batched_kernels,batch_index,batched_data):
    """ Predicts SVM across batch of data; saving accuracy, recall, precision
    """
    batched_predictions={}

    counter=0
    for batch,trained_svm in batched_kernels.items():
        
        # evaluate kernel on subsequent batch
        next_batch=batch_index[counter+1]

        # get data
        X=batched_data[next_batch]["features"]
        true=batched_data[next_batch]["outcomes"]
        
       
        # predict svm
        y_pred=trained_svm.predict(X)

        # compute accuracy
        accuracy=accuracy_score(true,y_pred)
        
        # compute recall
        recall=recall_score(true,y_pred)

        # compute precision
        precision=precision_score(true,y_pred)

        evaluation_dict={"accuracy":accuracy,"recall":recall,"precision":precision}

        # save predictions
        batched_predictions[batch]=evaluation_dict

    return batched_predictions


