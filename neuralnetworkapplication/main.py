from tkinter import *
from tkinter import ttk
import CreateNeuralNetwork as cnn


#create Tk object called root- the main application window
root=Tk()

#pass in the main application window into the CreateNeuralNetwork class constructor
#to create the widgets that will run in the main application window and get 
#access to them in the constructor
CreateNeuralNetworkApp=cnn.CreateNeuralNetwork(root)


#Tell tk to enter its event loop
root.mainloop()

