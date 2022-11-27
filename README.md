### Repo for Inference in Multiple Kernel Support Vector Machines

**Project Overview**
- Train Support Vector Machines for short-term trend classification on price data

**Available Data**
- Jan 2020 of high-frequency AAPL Quote Data from NYSE
- Original observations consist of 11 million quotes
- Applied standard techniques for cleaning and preprocessing 

**Feature Generation**
- Generates features shown to be relevant to the limit order book

**Model Formulation**
- Trains SVM via optimization of the dual formulation of objective
- Allows for a single kernel function or composition of multiple kernel functions
- Built off of SciKit Learn's SVM functionality 
- Multi-kernel optimization problem utilizes gradient descent algorithm 

  
**Repo Outline**
  - Feature Generation from Original Data
  - Kernel (Single and Multiple) Functions
