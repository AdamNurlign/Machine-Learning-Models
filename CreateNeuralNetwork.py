from tkinter import *
from tkinter import ttk
import numpy as np
import MLApp as ml
import pandas as pd
import threading
np.set_printoptions(suppress=True, precision=3) #no scientific notation, 3 decimals

class CreateNeuralNetwork:
    def __init__(self,root):
        #Root is the main application window widget
        root.title("Create Nueral Network")

        #self.mainframe is the main frame of the application
        self.mainframe=ttk.Frame(root,padding="3 3 12 12")
        self.mainframe.grid(row=0,column=0,sticky=(N,W,E,S))
        self.mainframe.rowconfigure(0,weight=1)
        self.mainframe.columnconfigure(0,weight=1)

        #dataset entry widget and its associated String Variable
        self.dataset=StringVar()
        self.dataset_entry=ttk.Entry(self.mainframe,textvariable=self.dataset)
        self.dataset_entry.grid(row=1,column=2,sticky=(N,W,E,S))

        #number of layers entry widget and its associated String Variable
        self.num_layers=StringVar()
        self.num_layers_entry=ttk.Entry(self.mainframe,textvariable=self.num_layers)
        self.num_layers_entry.grid(row=2,column=2,sticky=(N,W,E,S))
        
        self.num_layers_int=None

        #In this list I have access, in order, to the activation selection for each layer
        #This is a list of String Variables
        self.activation_selected_list=[]

        #In this list I have access, in order to the number of hidden layers selected 
        #for each layer. This is a list of String Variables.
        self.hidden_units_selected_list=[]



        #This loop also places the entry for writing the number of hidden units for 
        #each layer. It also collects each selection.

        #These are all the widgets that are related to layers (radiobuttons for selecting
        #activation function for each layer, and text entries for selecting the number
        #of hidden units in each layer)
        self.layer_widgets_list=[]

        #Buttons (4)
        #These 4 buttons are the construct, train, predict, and set_num_layers buttons that run their commands
        self.construct_button=ttk.Button(self.mainframe,text="Construct",command=self.construct)

        self.train_button=ttk.Button(self.mainframe,text="Train",command=self.start_training)

        self.predict_button=ttk.Button(self.mainframe,text="Predict",command=self.predict)
    
        self.set_num_layers_button=ttk.Button(self.mainframe,text="Set # of Layers",command=self.set_num_layers)
        self.set_num_layers_button.grid(row=5,column=1,sticky=(N,W,E,S))


        #Prediction argument entry widget and its corresponding String Variable
        self.prediction_arg=StringVar()
        self.prediction_arg_entry=ttk.Entry(self.mainframe,textvariable=self.prediction_arg)
        

        #Output of neural network label widget and its corresponding String Variable
        self.output=StringVar()
        self.output_label=ttk.Label(self.mainframe,textvariable=self.output)

        #Dynamic label widget and its associated String Variable which tells the 
        #user if the neural network has been constructed
        self.is_constructed=StringVar(value="Not constructed")
        self.is_constructed_label=ttk.Label(self.mainframe,textvariable=self.is_constructed)


        #Radiobutton widgets for selecting if the neural network will perform
        #regression or classification
        self.problem_type=StringVar()
        regression_radiobutton=ttk.Radiobutton(self.mainframe,text="regression",variable=self.problem_type,value="regression")
        classification_radiobutton=ttk.Radiobutton(self.mainframe,text="classification",variable=self.problem_type,value="classification")
        regression_radiobutton.grid(row=3,column=1,sticky=(N,W,E,S))
        classification_radiobutton.grid(row=4,column=1,sticky=(N,W,E,S))

        #Radiobutton widgets for selecting the loss function that the neural network
        #will perform
        self.loss_function_selection=StringVar()
        self.se_loss=ttk.Radiobutton(self.mainframe,text="Squared-error",variable=self.loss_function_selection,value="se")
        self.ce_loss=ttk.Radiobutton(self.mainframe,text="Cross-entropy",variable=self.loss_function_selection,value="ce")


        #Learning rate and number of epochs entry widgets and their associated
        #String Variables
        self.learning_rate=StringVar()
        self.num_epochs=StringVar()
        self.learning_rate_entry=ttk.Entry(self.mainframe,textvariable=self.learning_rate)
        self.num_epochs_entry=ttk.Entry(self.mainframe,textvariable=self.num_epochs)



        #Various static labels
        self.upload_dataset_label=ttk.Label(self.mainframe,text="Upload dataset (.csv file)")
        self.upload_dataset_label.grid(row=1,column=1,sticky=(N,W,E,S))

        self.num_layers_label=ttk.Label(self.mainframe,text="How many linear/fc layers?")
        self.num_layers_label.grid(row=2,column=1,sticky=(N,W,E,S))

        self.output_label_static=ttk.Label(self.mainframe,text="Output")

        self.loss_function_label=ttk.Label(self.mainframe,text="Loss function")

        self.learning_rate_label=ttk.Label(self.mainframe,text="Learning rate")

        self.num_epochs_label=ttk.Label(self.mainframe,text="Number of epochs")

        #Application's neural netowrk 
        self.nn=None

        #Application's dataset as a numpy array
        self.dataset_numpy=None
        
        #Segments of the dataset that I need for training and testing
        self.datasetX=None
        self.datasetY=None
        self.datasetXTest=None
        self.datasetYTest=None

        self.dataset_train_mean=None 
        self.dataset_train_std=None



    def construct(self,*args):
        #Dataset we will train on is taken from the StringVariable linked
        #to dataset entry widget
        csv_file=self.dataset.get() 
        dataset_df=pd.read_csv(csv_file)

        #Convert the pandas dataframe to a numpy array
        self.dataset_numpy=dataset_df.to_numpy()

        #The problem type is taken from a String variable linked to  
        #our 2 problem type radiobuttons (regression and classification buttons)
        problem_type=self.problem_type.get()
        if (problem_type=="regression"):
            self.datasetX,self.datasetY,self.datasetXTest,self.datasetYTest=ml.extractDatasetComponentsRegression(self.dataset_numpy)
        elif (problem_type=="classification"):
            self.dataset_numpy=self.dataset_numpy[1:,:]
            self.datasetX,self.datasetY,self.datasetXTest,self.datasetYTest=ml.extractDatasetComponentsClassification(self.dataset_numpy)
            self.datasetX=self.datasetX.astype(float)
            self.datasetY=self.datasetY.astype(float)
            self.datasetXTest=self.datasetXTest.astype(float)
            self.datasetYTest=self.datasetYTest.astype(float)

        #Normalization procedures for our dataset where in each attribute/feature we subtract the 
        #mean and divide by the standard deviation
        self.datasetX, dataset_train_mean, dataset_train_std = ml.standardize_data(self.datasetX)

        self.dataset_train_mean=dataset_train_mean
        self.dataset_train_std=dataset_train_std

        self.datasetXTest=(self.datasetXTest - dataset_train_mean) / dataset_train_std

        #Number of features/attributes in the dataset
        num_features=self.datasetX.shape[1]

        #Here is an example of a valid List of Layers that we can pass into ml.NeuralNetwork constructor:
        #ListOfLayersConcreteRegression=[ml.LinearLayer(8,3,activation="relu"),ml.LinearLayer(3,3,activation="relu"),ml.LinearLayer(3,3,activation="relu"),ml.LinearLayer(3,1,loss="se")]

        #int(self.num_layers.get())  is the number of linear layers the user selected for
        #self.hidden_units_selected_list of the form [StrVar("4"),StrVar("3"),StrVar("3")...] 
        #self.activation_selected_list of the form [StrVar("relu"),StrVar("sigmoid"),StrVar("softmax")]
        #self.loss_function_selection of the form StrVar("ce") or StrVar("se")
        #This should be enough to construct our nueral network

        layers=[]

        for i in range(self.num_layers_int):
            num_units=int(self.hidden_units_selected_list[i].get())
            activation_selected=self.activation_selected_list[i].get()
            if activation_selected=="none":
                activation_selected= None
            #At this point activation_selected is a string equal to 
            #"relu","sigmoid","softmax", or None
            if (i==0):
                #For the first linear layer the length of input going into it
                #is equal to the number of features in the dataset
                layers.append(ml.LinearLayer(num_features,num_units,activation=activation_selected))
            elif(i==self.num_layers_int-1):
                #For the last Linear Layer we need to specify the loss function
                #that corresponds to the enture neural network
                loss_fn=self.loss_function_selection.get()
                layers.append(ml.LinearLayer(int(self.hidden_units_selected_list[i-1].get()),num_units,activation=activation_selected,loss=loss_fn))
            else:
                layers.append(ml.LinearLayer(int(self.hidden_units_selected_list[i-1].get()),num_units,activation=activation_selected))

        

        self.nn=ml.NeuralNetwork(layers)
        self.is_constructed.set("Is Constructed")
        


    def start_training(self):
        threading.Thread(target=self.train,daemon=True).start()

    def train(self,*args):
        num_epochs=int(self.num_epochs.get())
        lr=float(self.learning_rate.get())
        self.nn.train_sgd(self.datasetX,self.datasetY,num_epochs,lr)
        self.is_constructed.set("Is Trained")
  
    def predict(self,*args):
        argument=np.fromstring(self.prediction_arg.get(),dtype=float,sep=',')
        argument=(argument-self.dataset_train_mean)/self.dataset_train_std
        prediction_value=self.nn.predict(argument)
        if (self.problem_type.get()=="regression"):
            self.output.set(str(prediction_value[0][0]))
        elif (self.problem_type.get()=="classification"):
            flattened_prediction=prediction_value.flatten()
            self.output.set(flattened_prediction)

    def set_num_layers(self,*args):
        #Note that this routine should get called when set_num_layers button gets
        #pressed AFTER the number of layers get entered.
        num_layers_int=int(self.num_layers.get())
        self.num_layers_int=num_layers_int

        #This loop destroys all widgets that configure layers
        for widget in self.layer_widgets_list:
            widget.destroy()
        self.layer_widgets_list.clear()

        #This clearing clears the list of StringVars attached to the widgets
        #which configure our layers. There is no destroy function for StringVars,
        #but because we destroy the lists that contain them and the widgets that
        #link to them, these old String Vars get garbage collected, saving space.
        self.activation_selected_list.clear()
        self.hidden_units_selected_list.clear()
        

        #This loop creates and  places the radiobutton widgets for selecting the activation 
        #function for each layer, and the text entry widgets for selecting the number
        #of hidden units in each layer. It also creates and places some additional 
        #layer widgets. It also collects all the selections into a list
        #self.layer_wigets_list.
        for i in range(num_layers_int):
            activation_label=ttk.Label(self.mainframe,text="Activation for linear layer "+str(i+1))
            activation_label.grid(row=4*i,column=3,sticky=(N,W,E,S))

            hidden_units_label=ttk.Label(self.mainframe,text="# of hidden units for linear layer "+str(i+1))
            hidden_units_label.grid(row=i,column=5,sticky=(N,W,E,S))

            activation_selection=StringVar()
            self.activation_selected_list.append(activation_selection)

            relu_button=ttk.Radiobutton(self.mainframe,text="relu",variable=activation_selection,value="relu")
            sigmoid_button=ttk.Radiobutton(self.mainframe,text="sigmoid",variable=activation_selection,value="sigmoid")
            softmax_button=ttk.Radiobutton(self.mainframe,text="softmax",variable=activation_selection,value="softmax")
            none_button=ttk.Radiobutton(self.mainframe,text="none",variable=activation_selection,value="none")
            relu_button.grid(row=4*i,column=4,sticky=(N,W,E,S))
            sigmoid_button.grid(row=4*i+1,column=4,sticky=(N,W,E,S))
            softmax_button.grid(row=4*i+2,column=4,sticky=(N,W,E,S))
            none_button.grid(row=4*i+3,column=4,sticky=(N,W,E,S))

            hidden_units_selected=StringVar()
            self.hidden_units_selected_list.append(hidden_units_selected)
            hidden_units_entry=ttk.Entry(self.mainframe,textvariable=hidden_units_selected)
            hidden_units_entry.grid(row=i,column=6,sticky=(N,W,E,S))
            self.layer_widgets_list.extend([activation_label,hidden_units_label,relu_button,sigmoid_button,softmax_button,none_button,hidden_units_entry])

            #For all of the widgets and StringVars in the right section of our GUI
            #we wait until the set_num_layers routine is triggered to grid them
            #so tkinter will respect our placement calculations.
            self.loss_function_label.grid(row=1,column=7,sticky=(N,W,E,S))
            self.learning_rate_label.grid(row=3,column=7,sticky=(N,W,E,S))
            self.num_epochs_label.grid(row=4,column=7,sticky=(N,W,E,S))
            self.output_label_static.grid(row=8,column=7,sticky=(N,W,E,S))
            self.output_label.grid(row=8,column=8,sticky=(N,W,E,S))
            self.is_constructed_label.grid(row=5,column=7,sticky=(N,W,E,S))
            self.construct_button.grid(row=6,column=7,sticky=(N,W,E,S))
            self.train_button.grid(row=6,column=8,sticky=(N,W,E,S))
            self.predict_button.grid(row=7,column=7,sticky=(N,W,E,S))
            self.se_loss.grid(row=1,column=8,sticky=(N,W,E,S))
            self.ce_loss.grid(row=2,column=8,sticky=(N,W,E,S))
            self.learning_rate_entry.grid(row=3,column=8,sticky=(N,W,E,S))
            self.num_epochs_entry.grid(row=4,column=8,sticky=(N,W,E,S))
            self.prediction_arg_entry.grid(row=7,column=8,sticky=(N,W,E,S))









        





