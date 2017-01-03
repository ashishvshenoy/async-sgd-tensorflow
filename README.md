# async-sgd-tensorflow
This tensorflow python program runs Logistic Regression Stochastic Gradient Descent Algorithm on the input dataset that is spread across 5 VMs in an asynchronous manner.

## Dataset
The dataset is the one used for the [Kaggle Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) sponsored by Criteo Labs. It is available [here](http://pages.cs.wisc.edu/~ashenoy/CS838/).

##Methodology Used
* 5 different processes are spawned on every VM in the cluster, but the graph is initialized only once by the process in VM-4-1. This allows us to maintain a global state for the gradients.
* The gradient is updated in an asynchronous manner since the iterations proceed independently without waiting for all the VMs to update the global gradient.
* An error_rate is calculated in the process running on VM-4-1 after every 1000 iterations. This approximates to around 2500 training iterations if we consider the training samples

## Environment
A 5 node cluster, each node with 20GB RAM and 4 cores was used to run this application.
