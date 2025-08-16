"""
Adam Nurlign 7/2/2025

Hello there! In this notebook I will be implementing a comprehensive Neural Network Library
in which you can build and customize your own neural networks which can be trained and evaluated on datasets of your choosing.

There are many libraries in Python such as PyTorch and Scikit-learn that give you access to machine learning modules and
allow you to build your own models made up of layers. I thought it would be a good exercise to be able to implement some of these
modules from scratch. I hope you enjoy!

Here are some current constraints to my Machine Learning modules:

-Can only perform stochastic gradient descent

"""

import numpy as np
import pandas as pd

#Layer Superclass
class Layer():
  def __init__(self):
    pass
  def apply(self,x):
    pass

#Linear Layer class
class LinearLayer(Layer):
  def __init__(self,in_dim,out_dim,activation=None,loss=None):
    super().__init__()
    """
    Each linear layer is defined by its weight matrix and bias vector which get applied to the input
    vector into the layer

    We should have a weight matrix with the dimension out_dim by in_dim which draws each element 
    from the normal distribution with mean 0 and variance 4/(in_dim+out_dim).
    """
    variance = 4/(in_dim+out_dim)
    std_dev=np.sqrt(variance)
    self.weights = np.random.normal(loc=0.0, scale=std_dev, size=(out_dim, in_dim))

    #The bias vector can be initialized with zeros
    self.bias=np.zeros((out_dim,1))

    #storing the derivative of objective with respect to this layer's weight matrix
    self.weightGrad=None
    #storing the deravitive of objective with respect to this layer's bias vector
    self.biasGrad=None
    #storing the deravitive of objective with respect to this layer's output (linear output z)
    self.outputGrad=None

    #storing the forward pass output value of this layer
    self.outputValue=None

    #eventually becomes the activation layer object for this layer
    self.activation=None

    #loss function either "ce" for cross-entropy of "se" for squared error
    self.loss=loss

    if (activation=="sigmoid"):
      self.activation=Sigmoid()
    elif (activation=="relu"):
      self.activation=reLU()
    elif (activation=="softmax"):
      self.activation=Softmax()
    else:
      pass


  def apply(self,x):
    return self.weights@x+self.bias
  
class reLU(Layer):
  def __init__(self):
    super().__init__()
    self.outputValue=None
  def apply(self,x):
    return np.maximum(x,0)
  
class Sigmoid(Layer):
  def __init__(self):
    super().__init__()
    self.outputValue=None
  def apply(self,x):
    return 1/(1+np.exp(-x))
  
class Softmax(Layer):
  def __init__(self):
    super().__init__()
    self.outputValue=None
  def apply(self, x):
    x=x.flatten()
    shift_x=x-np.max(x)
    exponent=np.exp(shift_x)
    denominator=np.sum(exponent)
    answer=exponent/denominator
    return answer.reshape(-1,1)

"""
This function calculates the deravitive of an activation layer output with respect to its corresponding linear layer output.
Note that the outputVal argument could be the linear layer output or activation layer output depending on how
the activation derative turns out to be.
"""
def derivOfActWrtLinearInput(activationType,outputVal):
  #For context, the deravitive of the objective with respect to linear output depends on 
  #derivative of activation with respect to linear output which depends on the activation function
  if (activationType=="sigmoid"):
    #For the sigmoid function the deravitive with respect to input is output * (1-output)
    #outputVal in this case should be the activation output
    fi=np.copy(outputVal)
    oneMinusfi=np.ones(fi.shape)-fi
    #This performs an element wise multiplication between the linear output and one minus the linear output
    derivOfAct=fi*oneMinusfi 
    #This turns the previous vector into a diagonal matrix. This is necessary because the derative of a vector
    #valued activation with respect to a vector valued linear output should be a matrix. Since sigmoid
    #is a element wise operation, this matrix is diagonal (0's on the non diagonal).
    derivOfActMatrix=np.diag(derivOfAct) 
    return derivOfActMatrix
  elif (activationType=="relu"):
    #The derivative of the relu activation function with respect to input (linear output) is a diagonal matrix of 
    #1's in the diagonal entries where the linear output is >0 and 0 elsewhere.
    #In this calse outputVal is the linear layer output.
    fi=np.copy(outputVal)
    #Takes the linear output and makes a corresponding vector with 1's in the positive places and 0 elsewhere
    OnesAndZeroes=(fi>0).astype(int)
    OnesAndZeroes=OnesAndZeroes.flatten()
    #Gets the diagonal matrix that constitutes the deravitive
    OnesAndZeroesMatrix=np.diag(OnesAndZeroes)
    return OnesAndZeroesMatrix
  elif (activationType=="softmax"):
    #Note here  output val will be the activation layer's output
    h=outputVal.reshape(-1,1)
    """
    The derivative of softmax activation with respect to linear input is very simple mathematically.
    The line of code below is an elegant, yet complicated expression which does the following operation: 
    on the diagonals i of the matrix returned do hi*1-hi where hi is the ith component of the activation output.
    On the off-diagonal entries representing the deravitive of activation component i with respect to linear output component j do
    -hi*hj. The matrix we return is symmetric meaning its transpose equals itself, so we don't have to worry about whether
    the result is in denominator or numerator format.
    """
    return np.diagflat(h)-np.dot(h,h.T)
  else:
    pass
    
#Neural Network Class- A neural network in this library is simply made up of a list of Layers
class NeuralNetwork():
  def __init__(self,layers):
    self.layers=layers

  def predict(self,x,storeValues=False):
    val=np.copy(x)
    if (val.ndim==1):
      val=val.reshape((val.shape[0],1))

    #In order to make a prediction in our neural network on an input vector x, we loop through
    #all the layers of the network and apply the layer to the input, and the layer's activation
    #if it has one. In order.
    for layer in self.layers:
      val=layer.apply(val)
      if (storeValues==True):
        #We will only be storing intermediate values of the forward pass in the case where we
        #are training our network.
        layer.outputValue=val
      if (layer.activation!=None):
        activationLayer=layer.activation
        val=activationLayer.apply(val)
        if (storeValues==True):
          #We will only be storing intermediate values of the forward pass in the case where we
          #are training our network.
          activationLayer.outputValue=val
    return val

  """
  This function specifically uses Stochastic Gradient Descent to train the neural network. 
  The steps of gradient descent in general is to perform the forward pass (storing intermediate values), 
  then perform the backward pass #1 to obtain the gradient of objective with respect to each linear output, 
  then perform the backward pass #2 to obtain the gradient of objective with respect to each weight matrix 
  and bias vector.
  """
  def train_sgd(self,train_x,train_y,num_epochs,learning_rate,batch_size=1):
    for epoch in range(num_epochs):
      #Shuffles the training dataset each epoch which is important for training
      perm = np.random.permutation(train_x.shape[0])
      train_x_shuffled = train_x[perm]
      train_y_shuffled = train_y[perm]
      #In stochastic gradient descent the batch size is 1 so each iteration we only fine-tune
      #the parameters based off a single training datapoint
      for i in range(len(train_x_shuffled)):
        x=train_x_shuffled[i].reshape(-1,1)
        y=train_y_shuffled[i].reshape(-1,1)
        #Forward Pass: While we perform the forward pass, in each layer we store the output 
        #of the layer so we have access to all the intermediate values.
        yHat=self.predict(x,storeValues=True)


        #Backward Pass #1: to get the gradient of objective with respect to each linear output

        for i in range(len(self.layers)-1,-1,-1):
          layer=self.layers[i]
          if (i==len(self.layers)-1):
            #Case that we are dealing with the output layer. In general dj/dz=da/dz*dJ/da 
            #where a is the final returned value of the neural network (yHat)
            activationType=None
            if isinstance(layer.activation, Sigmoid):
              activationType="sigmoid"
            elif isinstance(layer.activation, reLU):
              activationType="relu"
            elif isinstance(layer.activation, Softmax):
              activationType="softmax"
            else:
              pass

            #In the ouput layer the derative of objective with respect to linear output
            #depends on the loss function- squared-error or cross-entropy
            if (layer.loss=="se"):
              #Squared error loss function case
              if (activationType==None):
                #If the output layer has no activation (a=z) then da/dz=1.
                #dJ/da is really simple when the loss funciton is (yHat-y)^2.
                layer.outputGrad=2*(yHat-y)
              else:
                #Case where the output layer does have an activation function.
                djda=2*(yHat-y)
                dadz=None
                if (activationType=="softmax"):
                  dadz=derivOfActWrtLinearInput(activationType,yHat)
                elif(activationType=="relu"):
                  dadz=derivOfActWrtLinearInput(activationType,layer.outputValue)
                elif(activationType=="sigmoid"):
                  dadz=derivOfActWrtLinearInput(activationType,yHat)
                else:
                  pass
                layer.outputGrad=dadz@djda

            elif (layer.loss=="ce"):
              #Cross entropy loss function case
              if (activationType=="softmax"):
                #After complicated and long math it turns out in 
                #this special case where the loss function is cross-entropy and 
                #the final activation is softmax, that dJ/dz=da/dz*dJ/da=yHat-y
                self.layers[i].outputGrad=yHat-y

              else:
                """
                The cross-entropy loss output is essentially the negative log of the
                correct class component of yHat (the probability our neural network
                places on the correct class). Therefore the deravitive of cross entropy loss
                with respect to yHat=activation output is a vector of 0's in all components
                except for the correct class's component. In this component k we put -1/yHatk
                because the deravitive of log(x) is 1/x.
                """
                deriv=np.zeros_like(yHat)
                correctClassIndex=np.argmax(y)
                probCorrectClass=yHat[correctClassIndex,0]
                deriv[correctClassIndex,0]=-1*(1/(probCorrectClass)) 
                djda=deriv
                dadz=None
                if (activationType==None):
                  #In the case where there is no activation (a=z) da/dz=1
                  layer.outputGrad=djda
                else:
                  #We took care of the softmax case so we can only have an activation
                  #of relu or sigmoid. This calculates da/dz.
                  if (activationType=="relu"):
                    dadz=derivOfActWrtLinearInput(activationType,layer.outputValue)
                  elif (activationType=="sigmoid"):
                    dadz=derivOfActWrtLinearInput(activationType,yHat)
                  else:
                    pass
                  layer.outputGrad=dadz@djda

            continue

          else:

            """
            This is the case where we are not dealing with the output layer.
            For this case, in general  dj/dz= da/dz * dznext/da *dJ/dznext
            We loop starting from the last layer (output layer) and going backwards
            so for each iteration dJ/dznext will aready be calculated. All we need
            to calcualte in each iteration is da/dz (which we have a function for)
            and dznext/da, which mathematically is very clearly the next layer's 
            weight matrix.
            """
            #term1 is da/dz
            term1=None
            if (layer.activation==None):
              #When there is no activation (a=z) for da/dz we simply create the identity matrix
              term1=np.eye(layer.outputValue.shape[0])

            elif isinstance(layer.activation, Sigmoid):
              term1=derivOfActWrtLinearInput("sigmoid",layer.activation.outputValue.flatten())
            elif isinstance(layer.activation,reLU):
              term1=derivOfActWrtLinearInput("relu",layer.outputValue.flatten())
            elif isinstance(layer.activation,Softmax):
              term1=derivOfActWrtLinearInput("softmax",layer.activation.outputValue.flatten())
            else:
              pass

            #term2 is dznext/da=transpose(Wnext)
            term2=np.transpose(self.layers[i+1].weights)

            #term3 is dJ/dznext which we calucated in the previous iteration of this backward pass #1
            term3=self.layers[i+1].outputGrad

            #Calculate and store dj/dz
            self.layers[i].outputGrad=term1@term2@term3

   
        for i in range(len(self.layers)-1,-1,-1):
          """
          Backward Pass #2:
          For this part of SGD we calculate the gradient of the objective with respect to the weight matrices
          and bias vectors. We use this to update the weights and biases via the gradient descent equation.

          Here are the key equations: 
          dJ/dW= dz/dW * dJ/dz
          dJ/db= dz/db * dJ/dz
                
          Note that each dJ/dz was calculated in backward pass #1.
          dz/dW= the activation output of previous layer/ input into this linear layer
          dz/db=1
          """
          layer=self.layers[i]
          derivObjWrtBias=layer.outputGrad #dj/dz
          layer.bias=layer.bias-learning_rate*derivObjWrtBias

          #Special case for the first linear layer
          derivObjWrtWeights=None
          if i==0:
              derivObjWrtWeights=layer.outputGrad@np.transpose(x) #dz/dW=tranpose(x) for first linear layer
          else:
              derivObjWrtWeights=layer.outputGrad@np.transpose(self.layers[i-1].activation.outputValue) #dz/dW=transpose(a_previous)

          layer.weights=layer.weights-learning_rate*derivObjWrtWeights

def squared_error_loss(yHat,y):
  #In mathematics the squared error loss function is the sum of the squares of the difference between the two vectors
  difference=yHat-y
  return (np.transpose(difference))@difference

def absolute_error_loss(yHat,y):
  #In mathematics the absolute error loss function is the sum of the absolute values of the differences between the two cectors
  difference=yHat-y
  error_vector=np.abs(difference)
  return np.sum(error_vector)

def cross_entropy_loss(yHat,y):
  #In mathematics the cross entropy loss function is the sum of the probabilites put on the correct class for each prediction
  #Only one component of y will be 1 and the rest will be 0. Therefore to find the index i where yi=1 we can take the argmax
  oneIndex=np.argmax(y,axis=1)
  #This will be a column vector of the prob we predicted for the correct class for each datapoint
  probCorrectClass = yHat[np.arange(yHat.shape[0]), oneIndex]
  #vector of losses
  individualLosses=-1*np.log(probCorrectClass)
  return np.sum(individualLosses)

def numerize_output(Y):
  #This function is going to take an output column of strings (classes) and convert it into an output column of numbers from 0 to k-1
  Y_flat=Y.flatten()
  num_points=Y_flat.shape[0]
  #this is a 1d numpy array
  unique_elems=np.unique(Y_flat)
  unique_elems_list=list(unique_elems)
  num_unique_elems=len(unique_elems_list)
  k=num_unique_elems
  unique_elems_list_enum=enumerate(unique_elems_list)

  class_string_to_int_dict={string_class: int_class for (int_class,string_class) in unique_elems_list_enum}

  def string_to_int_class(string):
    #gets you the integer class
    return class_string_to_int_dict[string]

  convert_strings_to_ints=np.vectorize(string_to_int_class)
  y_numbers=convert_strings_to_ints(Y_flat)

  return y_numbers,k
  

def transform_classification_y_dataset(Y_train,Y_test,k):
  # Y_train and Y_test are going to have the dimensions of some nx1, we want to turn this into nxk where k is the number of classes, 
  #the classes go from 0 to k-1
  Y_train_flat=Y_train.flatten()
  Y_test_flat=Y_test.flatten()

  num_train_points=Y_train_flat.shape[0]
  num_test_points=Y_test_flat.shape[0]

  transformed_y_train=np.zeros((num_train_points,k))
  transformed_y_test=np.zeros((num_test_points,k))

  transformed_y_train[np.arange(num_train_points),Y_train_flat]=1
  transformed_y_test[np.arange(num_test_points),Y_test_flat]=1

  return transformed_y_train,transformed_y_test

def extractDatasetComponentsClassification(data):
  np.random.shuffle(data)
  splitIndex=int(0.8*len(data))

  Y=data[:,-1]
  data[:,-1],k=numerize_output(Y)

  X_train_data=data[:splitIndex,:-1]
  Y_train_data=data[:splitIndex,-1].reshape(-1,1).astype(int)

  X_test_data=data[splitIndex:,:-1]
  Y_test_data=data[splitIndex:,-1].reshape(-1,1).astype(int)

  Y_train_data,Y_test_data=transform_classification_y_dataset(Y_train_data,Y_test_data,k)

  return X_train_data,Y_train_data,X_test_data,Y_test_data

def extractDatasetComponentsRegression(data):
  np.random.shuffle(data)
  splitIndex=int(0.8*len(data))
  X_train_data=data[:splitIndex,:-1]
  Y_train_data=data[:splitIndex,-1].reshape(-1,1)
  X_test_data=data[splitIndex:,:-1]
  Y_test_data=data[splitIndex:,-1].reshape(-1,1)
  return X_train_data,Y_train_data,X_test_data,Y_test_data

def standardize_data(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data-mean) /std,mean,std