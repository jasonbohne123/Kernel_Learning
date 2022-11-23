from helpers import *
import numpy as np
from kernel import Kernel

def primal_dual_opt(X,y,m,kernel_type,order,gap=10e-4,inner_tol=10e-1,weight_threshold=0.01,maxouter_iter=100,maxinner_iter=10 ,verbose=True):
    """ Computes the optimal weights for the MKL objective using primal dual method
    """
    
    duality_gap=1 # initialize duality gap
    C=0.01 # penalty constant for SVM
    y_outer=np.outer(y,y) # outer product of y for efficiency
    counter=0
    d_m=np.ones(m)/m # initialize weights
    D=np.ones(m) # initialize descent direction
    mu=0 # initialize index of weight to be updated
    line_search_steps=25 # number of steps for line search
    gamma_max=0 # initialize step size
    
    # initialize kernel types
    if kernel_type=='linear':
        kernel_list=[Kernel(kernel_type=kernel_type)for i in range(1,order+1) ]
    elif kernel_type=='polynomial':
        
        kernel_list=[Kernel(kernel_type,i) for i in range(1,order+1)]
    elif kernel_type=='gaussian':
        kernel_list=[Kernel(kernel_type,bandwidth=i) for i in np.linspace(0.1,1,m)] # gamma hyperparam 

    else:
        print("Not Valid Kernel Type")
        return
    
    # stopping criteria is duality gap
    while duality_gap>gap and gap>0:
        if counter>maxouter_iter:
            break
        counter+=1

        # compute svm objective
        J_d,duality_gap,alpha=compute_dual(X,y,kernel_list,d_m,C) 
        if verbose:
            print("Duality",duality_gap)

        # gradient wrt each kernel
        gradient_j=[compute_gradient(i,X,y_outer,alpha) for i in kernel_list] 
        if verbose:
            print("Gradient is ",gradient_j)
   

        # max element within d vector
        mu=np.argmax(d_m)
        grad_mu=gradient_j[mu]
        
        # computes normalized descent direction ; satisfies equality constraints 
        D=descent_direction(d_m,mu,gradient_j,grad_mu)
        norm_D=np.sqrt(D.dot(D))
        if norm_D==0.0:
            break
        D=D/norm_D
        if verbose:
            print("Descent Direction is ",D)
        
        # init descent direction update
        J_hat=0
        d_hat=d_m
        D_hat=D
        inner_iter=0
        
        ### Investigate zero gamma step size
        gamma_list=[]

        while J_hat+inner_tol<J_d or inner_iter<maxinner_iter:
            inner_iter+=1 
            
            # indices where descent direction is negative, 
            # if none reached local max, else update by gamma step
            nonzero_D=np.where(D_hat<0)[0]
            if len(nonzero_D)==0:
                gamma_max=0  
            else:
                gamma_max=np.min(-d_hat[nonzero_D]/D_hat[nonzero_D])

            gamma_list.append(gamma_max)
            D=D_hat
            d_m=d_hat
            d_hat=d_m+gamma_max*D
            d_hat[d_hat<weight_threshold]=0
                
            D_hat[mu]=descent_direction(d_hat,mu,gradient_j,grad_mu)[mu]
            J_hat=compute_dual(X,y,kernel_list,d_hat,C,compute_gap=False)
            
        # line search in descent direction
        gamma_max=np.max(gamma_list)  
        gamma_step=line_search(X,y,kernel_list,D,d_m,gamma_max,disc=line_search_steps)
        
       
        d_m=(d_m+gamma_step*D)
        if verbose:
            print("Gamma Max is ",gamma_max)
            print("Gamma Step is ",gamma_step)
        
        # normalize and drop threshold
        d_m[d_m<weight_threshold]=0
        d_m=d_m/np.sum(d_m)
        
        if verbose:
            print("Weights are ",d_m)
       
    if abs(duality_gap)<gap:
            print("Duality Gap Reached")
            return d_m,kernel_list
    else:
        print("Max Iterations Reached")
        return d_m,kernel_list