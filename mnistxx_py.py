#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class NeuralNetwork():
    def __init__(self,X,Y,nn_layers,n_iters=2000,alpha = 0.01,lambda_=10):
        self.X = X
        self.Y = Y
        self.nn_layers= nn_layers
        self.n_iters= n_iters
        self.alpha= alpha
        self.lambda_= lambda_
        self.params = self.random_weights_initialization(nn_layers)
        self.fp_cache = None
        self.cost = None
        
    def tanh(self,z):
        return np.tanh(z)
    def sigmoid(self,z):
        return 1 / (1+np.exp(-z))
    def sigmoid_deriv(self,z):# z --> sigmoid(Z)
        return z * (1-z)
    def tanh_deriv(self,z): # z --> tanh(Z)
        return 1 - (z)**2
    def relu(self,Z):
        z_copy = np.array(Z,copy=True)
        return np.maximum(0,z_copy)
    def relu_backward(self,dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0
        return dZ
    
    def random_weights_initialization(self,n_layers):
        params = {}

        # n_layers --> (len-1) is no of layers and elements of list are no of neurons in (index+1)th layer

        for i in range(1,len(n_layers)):
            params["W"+str(i)] = np.random.randn(n_layers[i-1],n_layers[i]) 
            params["b"+str(i)] = b3 = np.zeros((1,n_layers[i]))

        return params
    
    
    
    def forward_prop(self,input_X,parameters):   
        cache = {
            "A0":input_X
        }
        last_index = int(len(parameters)/2)
        for i in range(1,last_index):
            cache["Z"+str(i)] = np.dot(cache["A"+str(i-1)],parameters["W"+str(i)]) + parameters["b"+str(i)]
            cache["A"+str(i)] = self.relu(cache["Z"+str(i)])

        cache["Z"+str(last_index)] = np.dot(cache["A"+str(last_index-1)],
                                            parameters["W"+str(last_index)])  + parameters["b"+str(last_index)]
        cache["A"+str(last_index)] = self.sigmoid(cache["Z"+str(last_index)])
        self.fp_cache=cache
        return cache["A"+str(last_index)],cache
    
    
    def compute_cost(self,pred_output,actual_output,parameters,lamdba):    
        A = pred_output
        y =actual_output
        m = y.shape[0]
        norm=0
        for i in range(1,int(len(parameters)/2)):
            norm = np.linalg.norm(parameters["W"+str(i)])

        cost_ =  (1./m)* np.sum((A-y)**2) + (lamdba * norm)/(2*m)
        return np.squeeze(cost_)


    def get_accuracy_value(self,Y_hat, Y):
        testing = Y_hat.copy()
        testing[Y_hat>0.5]= 1 
        testing[Y_hat <= 0.5]=0
        return (testing == Y).all(axis=1).mean()
    
    def backward_prop(self,Y,cache,parameters):

        last_index = int(len(parameters)/2)
        m = Y.shape[0]
        back_cache = {}

        delta3  = 2*(cache["A"+str(last_index)] - Y) * self.sigmoid_deriv(cache["A"+str(last_index)])
        back_cache["dW"+str(last_index)] = np.dot(cache["A"+str(last_index-1)].T,delta3)/ m 
        back_cache["db"+str(last_index)] =np.sum(delta3,axis=0,keepdims=True)/ m 

        delta = delta3
        for i in reversed(range(1,last_index)):
            delta =self.relu_backward(np.dot(delta,parameters["W"+str(i+1)].T),cache["Z"+str(i)])
            back_cache["dW"+str(i)] = np.dot(cache["A"+str(i-1)].T,delta) / m
            back_cache["db"+str(i)] = np.sum(delta,axis=0,keepdims=True)/ m

        for i in range(1,(last_index)):
            assert (parameters["W"+str(i)].shape == back_cache["dW"+str(i)].shape) 

        return back_cache
    
    def update_params(self,params,backward_cache,learning_rate,lamdba,m):
        ferb = (learning_rate * lamdba)/m
        last_index = int(len(params)/2)
        for i in range(1,last_index+1):
            params["W"+str(i)] = (1-ferb)*params["W"+str(i)]- (learning_rate * backward_cache["dW"+str(i)])
            params["b"+str(i)] = params["b"+str(i)]- (learning_rate * backward_cache["db"+str(i)])
        return params
    
    def train(self,print_cost=False):    
        params  = self.random_weights_initialization(self.nn_layers)
        J_cost =[]
        print("Starting Training")
        for i in range(self.n_iters):

            A3,cache = self.forward_prop(self.X,params)
            J = self.compute_cost(A3,self.Y,params,self.lambda_)
            J_cost.append(J)
            if print_cost:
                if (i%100 == 0):
                    print("iter:",str(i),"cost:",str(J),"Accuracy:",str(self.get_accuracy_value(A3,self.Y)))
                    print("----------------------------------------------------------------------")

            back_cache = self.backward_prop(self.Y,cache,params)

            self.params  = self.update_params(params,back_cache,self.alpha,self.lambda_,self.X.shape[0])
        self.cost  = J_cost
        print("Training Done.")
        print("With \n" + "Cost: "+str(J)+" Accuracy: "+str(self.get_accuracy_value(A3,self.Y)))
        

    
    def predict(self,testX,threshold):    
        cache = {
            "A0":testX
        }
        last_index = int(len(self.params)/2)
        for i in range(1,last_index):
            cache["Z"+str(i)] = np.dot(cache["A"+str(i-1)],self.params["W"+str(i)]) + self.params["b"+str(i)]
            cache["A"+str(i)] = self.relu(cache["Z"+str(i)])

        cache["Z"+str(last_index)] = np.dot(cache["A"+str(last_index-1)],
                                            self.params["W"+str(last_index)])  + self.params["b"+str(last_index)]
        cache["A"+str(last_index)] = self.sigmoid(cache["Z"+str(last_index)])

        output = np.array(cache["A"+str(last_index)],copy=True)
        output[output > threshold] = 1
        output[output <= threshold] = 0    

        return output
    
    def evaluate(self,input_X,output_y,threshold=0.5):
        predictions = self.predict(input_X,threshold)
        print( " precision:" , str(precision_score(output_y,predictions))[:5],"\n",
        "recall:",str(recall_score(output_y,predictions))[:5],"\n",
        "fl score:",str(f1_score(output_y,predictions))[:5],"\n",
        "accuracy:",str((predictions == output_y).all(axis=1).mean())[:5])
        results = confusion_matrix(output_y,predictions)
        fig, ax = plt.subplots()
        ax.matshow(results)
        for (i, j), z in np.ndenumerate(results):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.show()
    
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost)),self.cost)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()


# In[3]:


import sys, numpy as np 
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Getting the Data for 1 and 0 to classify:

# In[4]:


y_1_train = np.where(y_train == 1)
y_0_train = np.where(y_train == 0)
index_train =np.append(y_1_train[0],y_0_train[0])
np.random.shuffle(index_train)

y_1_test = np.where(y_test == 1)
y_0_test = np.where(y_test == 0)
index_test =np.append(y_1_test[0],y_0_test[0])
np.random.shuffle(index_test)


# index
x_train= x_train[(index_train)]
y_train= y_train[(index_train)]
x_test= x_test[(index_test)]
y_test= y_test[(index_test)]

print(index_train)


# In[5]:


n_samples_train = 12000
i = np.random.randint(0, n_samples_train)
img = Image.fromarray(x_train[i])
plt.imshow(img)

y_train[i]


# In[6]:


images, labels = (x_train[0:n_samples_train].reshape(n_samples_train,28*28) /255, y_train[0:n_samples_train].reshape(n_samples_train,1)) 
df = pd.DataFrame(images)
images.shape,labels.shape


# In[20]:


# params_ , cost_ = train(images,labels,[images.shape[1],6,4,labels.shape[1]],n_iters=2000,alpha =  0.01,lambda_=10,)
nn_arch=[images.shape[1],6,5,labels.shape[1]]
nn =NeuralNetwork(images,labels,nn_arch)
nn.train()


# In[21]:


nn.plot_cost()


# Testing the Trained Neural Network

# In[22]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score ,f1_score


# In[23]:


n_test_samples = x_test.shape[0]
testX = x_test.reshape(n_test_samples,28*28)/255
testY = y_test.reshape(n_test_samples,1)


# In[24]:


thres = 0.5
print("Threshold:",str(thres))
print("-------------test set-------------")
nn.evaluate(testX,testY,thres)

print("-------------train set-------------")
nn.evaluate(images,labels,thres)


# In[25]:


thres = 0.8
print("Threshold:",str(thres))
print("-------------test set-------------")
nn.evaluate(testX,testY,thres)

print("-------------train set-------------")
nn.evaluate(images,labels,thres)


# In[26]:


thres = 0.3
print("Threshold:",str(thres))
print("-------------test set-------------")
nn.evaluate(testX,testY,thres)

print("-------------train set-------------")
nn.evaluate(images,labels,thres)


# In[27]:


thres = 0.7
print("Threshold:",str(thres))
print("-------------test set-------------")
nn.evaluate(testX,testY,thres)

print("-------------train set-------------")
nn.evaluate(images,labels,thres)


# In[28]:


thres = 0.9
print("Threshold:",str(thres))
print("-------------test set-------------")
nn.evaluate(testX,testY,thres)

print("-------------train set-------------")
nn.evaluate(images,labels,thres)

