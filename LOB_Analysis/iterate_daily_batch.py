
import sys
import pandas  as pd
from LOB_Analysis.batch_data import batch_features,batch_solve_mkl
from LOB_Analysis.batch_kernel import train_svm_batch, predict_svm_batch

def daily_batch(start,end,kernel_type, order, batch_size):
    """ Train a SVM in batches across the entire dataset.
    """
    
    all_accuracy={}
    all_recall={}
    dt_range = pd.date_range(start, end, freq='D')
    path='/home/jbohn/jupyter/personal/Kernel_Learning/'

    for dt in dt_range:
        try:
            labeled_data=pd.read_csv(path+"Features/Cleaned_Features/labeled_data_"+str(dt.date())+".csv",index_col=0)
        except Exception as e:
            print(e)
            print("No data for "+str(dt))
            continue
        
        features=labeled_data[['FB0','FA0','FB2','FA2']]
        outcomes=labeled_data['outcome']

        batched_data = batch_features(features,outcomes, batch_size)
        batch_estimates=batch_solve_mkl(features,outcomes,order,batch_size,kernel_type,order,verbose=False)
        batched_kernels, batch_index = train_svm_batch(batched_data, batch_estimates, kernel_type, order)
        batched_predictions = predict_svm_batch(batched_kernels, batch_index, batched_data, batch_estimates, kernel_type, order)

        accuracy=[i["accuracy"] for i in batched_predictions.values()]
        recall=[i["recall"] for i in batched_predictions.values()]
        intraday_dt=[i for i in batched_predictions.keys()]

        for i in range(len(accuracy)):
            all_accuracy[intraday_dt[i]]=accuracy[i]
            all_recall[intraday_dt[i]]=recall[i]
    

    return all_accuracy, all_recall