### EX NO : 06
### DATE  : 29/04/2022
# <p align="center"> ANN BY BACK PROPAGATION ALGORITHM </p>
## Aim:
   To implement multi layer artificial neural network using back propagation algorithm.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## Related Theory Concept:
The Backpropagation algorithm is a supervised learning method for multilayer feed-forward networks from the field of Artificial Neural Networks.

Feed-forward neural networks are inspired by the information processing of one or more neural cells, called a neuron. A neuron accepts input signals via its dendrites, which pass the electrical signal down to the cell body. The axon carries the signal out to synapses, which are the connections of a cell’s axon to other cell’s dendrites.

The principle of the backpropagation approach is to model a given function by modifying internal weightings of input signals to produce an expected output signal. The system is trained using a supervised learning method, where the error between the system’s output and a known expected output is presented to the system and used to modify its internal state.

Technically, the backpropagation algorithm is a method for training the weights in a multilayer feed-forward neural network. As such, it requires a network structure to be defined of one or more layers where one layer is fully connected to the next layer. A standard network structure is one input layer, one hidden layer, and one output layer.

## Algorithm
1.Import necessary libraries
2.Define the Sigmoid and derivative of sigmoid functions
3.Define the model and the epoch, input layer, output layer values
4.Run the model.

## Program:
```
/*
Program to implement ANN by back propagation algorithm.
Developed by   : P.Sanjay
RegisterNumber : 212220230042 
*/
```
```
import numpy as np
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
y=y/100
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivatives_sigmoid(x):
    return x*(1-x)
epoch=7000
lr=0.1
inputlayer_neurons=2
hiddenlayer_neurons=3
output_neurons=1
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    hinp1=np.dot(X,wh)
hinp=hinp1+bh
hlayer_act=sigmoid(hinp)
outinp1=np.dot(hlayer_act,wout)
outinp=outinp1+bout
output=sigmoid(outinp)
EO=y-output
outgrad=derivatives_sigmoid(output)
d_output=EO*outgrad
EH=d_output.dot(wout.T)
hiddengrad=derivatives_sigmoid(hlayer_act)
d_hiddenlayer=EH*hiddengrad
wout+=hlayer_act.T.dot(d_output)*lr
wh+=X.T.dot(d_hiddenlayer)*lr
print("Input: \n"+str(X))
print("\nActual Output: \n",str(y))
print("\nPredicted Output: \n",str(output))
```

## Output:
![image](https://user-images.githubusercontent.com/75235426/169488215-456a73e1-2640-42f0-8b3e-f1aa5035f14d.png)


## Result:
Thus the python program successully implemented multi layer artificial neural network using back propagation algorithm.
