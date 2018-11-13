
from sklearn.base import BaseEstimator
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
import numpy as np
import sys


np.random.seed(111)
torch.manual_seed(111)

class Net(torch.nn.Module):
    def __init__(self, n_feature, h_sizes, n_output,dropout = 0): 
        
        """
        Initialize a Neural network architecture
        
        Parameters
        ----------
        n_feature : int. Number of features, ex: 10
        
        h_sizes : list of sizes of hidden layers, ex: [2,4,5]
        
        n_output : default 1 for regression. 
        
        dropout : 
        """

        super(Net, self).__init__()
       
        if h_sizes is not None:
            assert type(h_sizes) == list, "h_sizes should be a list with sizes of hidden layers.";
            h_sizes = [n_feature] + h_sizes
            self.hidden = torch.nn.ModuleList()
            for k in range(len(h_sizes)-1):
                self.hidden.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
                self.hidden.append(torch.nn.Dropout(p= dropout))
                self.hidden.append(torch.nn.ReLU())
        else:
            h_sizes = [n_feature]
        self.predict = torch.nn.Linear(h_sizes[-1], n_output)   # output layer
        self.hidden.apply(self.init_weights)
        self.predict.apply(self.init_weights)

    def forward(self, x):
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
        x = self.predict(x)   # output layer
        return x

    def init_weights(self,m):
    	if type(m) == torch.nn.Linear:
        	torch.nn.init.xavier_uniform(m.weight)
        	m.bias.data.fill_(0.01)

class PytorchRegressor(BaseEstimator):
    def __init__(self,n_feature,h_sizes, batch_size=64, max_epoch = 10**5, tenacity = 5, lr = 0.001,use_cuda =False):
        """

        Scikit-Learn like Pytorch Regressor


        Parameters
        ----------
        
        n_feature : int, number of input features 

        h_sizes : list of integers
                Length of list is the number of hidden layers
                Elements of list are the number of neurons in each hidden layer
                ex: [4,5,6]  ~ three hidden layers with 4,5,6 neurons respectively 

        batch_size : int, default = 64

        max_epoch : int, upper limit of no. of epochs. Default = 10**5

        tenacity : int, default = 5 

        lr : learning rate, float. default = 0.001

        use_cuda : Boolean, default = False. Set to True to enable CUDA

        """
        self.n_feature = n_feature
        self.h_sizes = h_sizes
        self.net = Net(self.n_feature,self.h_sizes,1)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.net = self.net.cuda()
        self.max_epoch = max_epoch
        self.tenacity = tenacity  
        self.batch_size = batch_size
        self.lr = lr

        
        
        
    def _convert_to_tensor(self,X, variable = True):
        """
        Convert to FloatTensor

        Parameters
        ----------
        X : Pytorch tensor or Numpy ndarray or Pandas DataFrame
        variable :  Boolean, Default = True. Convert a tensor to variable 


        Returns
        -------
        X: Pytorch FloatTensor

        """
        if type(X) == torch.FloatTensor: # isinstance(X, torch.FloatTensor):
            tensor_ = X     
        elif type(X) == pd.DataFrame:
            tensor_ =  torch.from_numpy(X.values).type(torch.FloatTensor)
        else:
        	tensor_ =  torch.from_numpy(X).type(torch.FloatTensor)

        
        if len(tensor_.shape) <2:
            tensor_=  torch.unsqueeze(tensor_,1)
        
        if variable: 
            return Variable(tensor_) 
        return tensor_

    def _initialize_optimizer(self):
    	"""
    	Initialize the optimizer

    	Default : torch.optim.SGD()
    	
    	"""
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

    
    def _initialize_lossfunc(self):
    	"""
    	Initialize a loss function using method normalized_loss

    	"""
        #self.loss_func = torch.nn.MSELoss()
        self.loss_func = self.normalized_loss

    def normalized_loss(self,pred,y):
        """
        Define normalized loss

        Parameters
        ----------
        pred : predicted values from predict method

        y : actual values (ground truth)

        Returns
        -------

        L1Loss function

        """
    	l1 = torch.nn.L1Loss()
    	if torch.abs(y.data).mean() > 0:
    		return l1(pred,y)/torch.abs(y.data).mean()
    	else:
    		return l1(pred,y)
        
    def prepare_split(self, X, y, validation_data, validation_split):
        """
        Prepare validation data.
        
        Parameters
        ----------
        X : Pytorch tensor, shape (n_samples, n_features)
            Training data
        y : Pytorch tensor, shape (n_samples, n_output)
            Target values. Should be same as X's dtype.

        validation_split : float or optional. if float, should be a value between 0.00 and 1.00.
                           Default value is 0.25. 

        validation_data  : Pytorch tensor or Numpy ndarray or Pandas DataFrame, assign if an additional validation data is available.


        Assign value to either validation_split or validation_data
        
        """
        assert validation_split or validation_data, "Assign Value to either validation_split or validation_data" # Only one of the two can be none
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = self._convert_to_tensor(validation_data)
        elif validation_split is not None:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split*len(X)):]
            devidx = permutation[0:int(validation_split*len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]
        else:
            trainX,trainy = X,y
            devX,devy = None,None
        return trainX, trainy, devX, devy


    def fit(self,X,y,validation_data=None, validation_split=0.25, early_stop = True): 
    	"""
        Fit pytorch model.
        
        Parameters
        ----------
        X : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, n_features)
            Training data
        y : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, n_output)
            Target values. Should be same as X's dtype 

        validation_split : float or optional. if float, should be a value between 0.00 and 1.00.
                           Default value is 0.25. 

        validation_data : Pytorch tensor or Numpy ndarray or Pandas DataFrame, assign if an additional validation data is available.

        early_stop : Boolean, 

        Returns
        -------
        self: Returns an instance of self

        """
        self.X = self._convert_to_tensor(X)
        self.y = self._convert_to_tensor(y)
        if self.use_cuda:
            self.X = self.X.cuda()
            self.y = self.y.cuda()
        
        bestscore = sys.maxint
        self.nepoch = 0
        stop_train = False
        early_stop_count = 0
        
        self._initialize_optimizer()
        self._initialize_lossfunc()
        
        trainX, trainy, devX, devy = self.prepare_split(self.X, self.y, validation_data,
                                                        validation_split)
        if self.use_cuda:
            trainX, trainy, devX, devy = trainX.cuda(), trainy.cuda(), devX.cuda(), devy.cuda()
            
        self.all_losses = []

        print("Training the network\n Loss on validation:\n")

        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX,trainy)
            validation_prediction = self.net(devX); #print(validation_prediction) #Variable; 
            validation_loss = self.loss_func(validation_prediction,devy)#Variable
            print("Epoch "+str(self.nepoch) +" "+str(validation_loss.data.numpy()[0])+"\n")
            if validation_loss.data.numpy()[0] < bestscore:
                bestscore = validation_loss.data.numpy()[0]
                early_stop_count = 0
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1                                 
        return self 
                
    def trainepoch(self,X,y):
    	"""
    	Batch train a single epoch.

        Parameters
        ----------
        X : Pytorch tensor, shape (n_samples, n_features)
            Training data
        y : Pytorch tensor, shape (n_samples, n_output)
            Target values. Should be same as X's dtype.
    	"""
        idx_permutation = np.random.permutation(len(X))
        for i in range(0,len(X), self.batch_size):
            self.optimizer.zero_grad()
            idx = torch.LongTensor(idx_permutation[i:i + self.batch_size])
            idx = Variable(idx)
            if self.use_cuda:    
                idx = idx.cuda()
            X_batch = X.index_select(0, idx); 
            y_batch = y.index_select(0,idx)
            if self.use_cuda:
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
            prediction = self.net(X_batch) 
            loss = self.loss_func(prediction,y_batch)
            loss.backward() # backpropagation, compute gradients
            self.optimizer.step()
        self.nepoch += 1
        

    def predict(self,X):
        """
        Parameters: 
        ------------
        X : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, n_features)
            
        Returns
        ---------
        Numpy ndarray of predictions of X.

        """
        X = self._convert_to_tensor(X)
        return self.net(X).data.numpy()
    
    def score(self,X,y):
        """Returns the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, n_features)
        y : Pytorch tensor or Numpy ndarray or Pandas DataFrame, shape (n_samples, 1)
            True values for X.

        Returns
        -------
        score : float
            Adjusted R^2.
        """
        X = self._convert_to_tensor(X)
        y = self._convert_to_tensor(y)
        predicted = self.net(X)
        TSS = (y -y.mean()).pow(2).sum()
        RSS = (y -predicted).pow(2).sum()
        self.Rsq = 1-(RSS/TSS) # R-square
        self.Rsq = self.Rsq.data[0]
        return 1- ((1-self.Rsq)*(X.shape[0] -1))/(X.shape[0] -X.shape[1]-1) # adjusted R-square

