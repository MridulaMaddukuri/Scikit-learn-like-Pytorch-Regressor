## Scikit-learn like Pytorch Regressor


A pytorch regressor that mimics scikit-learn objects 


class PytorchRegressor(n_feature, h_sizes, batch_size=64, lr=0.001,
                        max_epoch = 10^5, tenacity = 5, use_cuda = False)

# Parameters

        n_feature : int, number of input features.
                    if X is the input data then n_feature = X.shape[0]

        h_sizes : list of integers
                Length of list is the number of hidden layers
                Elements of list are the number of neurons in each hidden layer
                ex: [4,5,6]  ~ three hidden layers with 4,5,6 neurons respectively 

        batch_size : int, default = 64

        max_epoch : int, upper limit of no. of epochs. Default = 10**5

        tenacity : int, default = 5 
                  Number of steps of validation loss increase before early stopping 

        lr : learning rate, float. default = 0.001

        use_cuda : Boolean, default = False. Set to True to enable CUDA

# Attributes





# Methods

fit
predict 
score
get_params





