### Repo for Inference in Multiple Kernel Support Vector Machines

**Project Overview**
- Train Support Vector Machines for short-term trend classification on price data

**Available Data**
- Jan 2020 of high-frequency AAPL Quote Data from NYSE
- Original observations consist of 11 million quotes
- Applied standard techniques for cleaning and preprocessing 

**Feature Generation**
- Generates features shown to be relevant to the limit order book
- Examples inclue orderbook imbalance, quoted size and change in quoted size, average weighted spread
- Features aggregated into 1-second increments by the sample mean

**Model Formulation**
- Trains SVM via optimization of the dual formulation of objective
- Allows for a single kernel function or composition of multiple kernel functions
- Built off of SciKit Learn's SVM functionality 
- Multi-kernel optimization problem utilizes gradient descent algorithm 

**Numerical Results**
- Given labels features; train SVM for both single and multiple gaussian and polynomial kernels
- Number of basis functions for kernels selected were 1, 3, 5
- Polynomial kernels generated in order of basis function size
- Gaussian kernels generated equally spaced bandwidth parameter
- Trained across batches; evaluated on subsequent batch
- Accuracy and Weighted Precision metrics returned

