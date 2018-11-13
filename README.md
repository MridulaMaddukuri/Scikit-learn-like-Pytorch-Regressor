# Scikit-learn like Pytorch Regressor


A pytorch regressor that mimics scikit-learn objects 


```python 
PR = PytorchRegressor(n_feature, h_sizes, batch_size=64, lr=0.001, max_epoch = 10^5, tenacity = 5, use_cuda = False)
 ```
### Parameters

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


### Methods

#### _fit_

```python 
PR.fit(X,y,validation_data=None, validation_split=0.25, early_stop = True)
 ```
    	  
        Fit pytorch model.
        
        Parameters
        ----------
        X : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, n_features)
            Training data
        y : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, n_output)
            Target values. Should be same as X's dtype 

        validation_split : float or optional. if float, should be a value between 0.00 and 1.00.
                           Default value is 0.25. Fraction of data to be used for validation (early stopping)

        validation_data : Pytorch tensor or Numpy ndarray or Pandas DataFrame, 
                          assign if an additional validation data is available.

        early_stop : Boolean, default = True 



#### _score_

    ```python PR.score(X,y) ```
        
        Returns the adjusted R^2 of the prediction.
        
        Parameters
        ----------
        X : Pytorch tensor, shape (*, n_features)
        y : Pytorch tensor, shape (*, 1)
            True values for X.

        Returns
        -------
        score : float
            Adjusted R^2.
    

#### _predict_ 


    ```python PR.predict(X)```
        
        Predict using the trained network
        
        Parameters: 
        ------------
        X : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, n_features)
            
        Returns
        ---------
        Numpy ndarray of predictions of X.


#### _get_params_
      
     get_params(deep = True)
     
        Get params of the regressor
        



### Attributes
     Rsq : R^2 of the model
     




