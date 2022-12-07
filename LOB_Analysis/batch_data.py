import numpy as np

from Kernels.mkl_solver import primal_dual_opt

def batch_features(features,outcomes,batch_size):
    """ Returns a batch of features and outcomes
    """
    batched_dict={}
    for i in range(0,len(features),batch_size):
        # save the features and outcomes for each batch; timestamped by last interval 
        batched_dict[i/batch_size]={"last_interval":features.index[i] ,"features":features[i:i+batch_size], "outcomes":outcomes[i:i+batch_size]}
    return batched_dict

def batch_solve_mkl(X,y,m,batch_size,kernel_type,order,gap=10e-2,inner_tol=10e-3,weight_threshold=0.01,maxouter_iter=100,maxinner_iter=10 ,batch_verbose=True,verbose=True):
    """ Solves the MKL problem for a batch of data
    """
    n=X.shape[0]
    batched_dict=batch_features(X,y,batch_size)
    batched_estimates=np.zeros((n,m))
    for i in range(0,n,batch_size):
        weights,kernel=primal_dual_opt(batched_dict[i/batch_size]["features"].values,batched_dict[i/batch_size]["outcomes"].values,m,kernel_type,order,gap,inner_tol,weight_threshold,maxouter_iter,maxinner_iter ,verbose)
        batched_estimates[i,:]=weights
        if batch_verbose:
            print("Batch ",i,"Last Interval", batched_dict[i/batch_size]["last_interval"], "complete with weights ",weights)
    return batched_estimates