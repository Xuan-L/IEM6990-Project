import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from math import *
import time
import random 
from numpy import linalg as LA
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Preload the mnist data as the applied dataset
mnist = fetch_openml('mnist_784', version=1)
U0, v0 = mnist["data"], mnist["target"]
U= U0.astype(np.double)
v = v0.astype(np.uint8)

v_bin_5_lst = [2*int(v[i]==5)-1 for i in range(len(v))]
df_U = pd.DataFrame(data=U)
df_v = pd.DataFrame(data=np.asarray(v_bin_5_lst),  columns=['label'])
df_data_merged =pd.concat([df_U, df_v.reindex(df_U.index)], axis=1)

# Or a random dataset (The parameter in following coding is applied based on this dataset)
n_attributes = 100
n_labels = 70000
np.random.seed(1)
mu, sigma = 1,1
rnd_data = np.random.normal(mu, sigma, size=[n_labels,n_attributes,])

df_rnd_data = pd.DataFrame(data=rnd_data )
df_rnd_labels = pd.DataFrame(data=np.asarray(v_bin_5_lst),  columns=['label'])
df_data_merged2 =pd.concat([df_rnd_data, df_rnd_labels.reindex(df_rnd_data.index)], axis=1)

# Define the function to split training and testing set
def split_train_test(df_data_merged, train_set_size,test_set_size,n):
    np.random.seed(0)
    shuffled_indices = np.random.permutation(len(df_data_merged))
    batch_size = int(train_set_size/n)
    dic_train_sets_indices= {}
    dic_train_sets = {}
    for i in range(n):
        dic_train_sets_indices[i] = shuffled_indices[i*batch_size:(i+1)*batch_size]
        dic_train_sets[i] = df_data_merged.iloc[dic_train_sets_indices[i]]
    dic_train_set = {}
    dic_train_set_indices = shuffled_indices[:n*batch_size]
    dic_train_set[0] = df_data_merged.iloc[dic_train_set_indices]
    dic_test_set= {}
    test_indices = shuffled_indices[-test_set_size:]
    dic_test_set[0] = df_data_merged.iloc[test_indices]
    return dic_train_set, dic_train_sets, dic_test_set

# Set number of variables in one block, d, number of blocks, b, and number of agents, n. 
d = 10
b = 10
n = 10

# Define the adjency matrix, phi and initial x
a = np.ones([n,n])/n
phi = np.ones((1,1))
x0 = np.ones((d,1))

# Define the objective function (Perfer logistic regression function), i.e., U
def f_obj(XXX):
    return(XXX)
# Define the gradient of the f based on one block
def local_grad_f(XXX):
    return(XXX)
# Define the local optimization function
def local_optim(XXX):
    return(XXX)
# Define the block sending rule (Use the agent and number of iterations to output the block it will sent)
def block_sending(XXX):
    return(XXX)
# Define the way to calculate new phi (Use block and a)
def new_phi(XXX):
    return(XXX)
    
# Define the main function
def Block_sonata(dic_train_set,max_iter,epoch_size,x0,step_size,mu_param,b,n):
    f_values = np.zeros(epoch_size+1)
    # Define the inital x for each block
    for i in range(b):
        locals()["x"+str(i)] = x0 + 0.0
    # Define the inital phi for each block
    for i in range(b):
        locals()["phi"+str(i)] = phi + 0.0
    # Define the inital gradient for each block
    for i in range(b):
        locals()["gradient"+str(i)] = local_grad_f#(locals()["x"+str(i)])
    for k in range(max_iter+1):
        #Update the step size
        step_size_new = step_size**(k+1)
        # There should have two loop, one is for n, another is for b. However, based on our block selection rule, we can combine these two as one loop
        for i in range(n):   
            #Get the block agent i will send in interation k
            p = block_sending
            #Update the x of block p
            locals()["x_new"+str(p)] = local_optim
            #Calculate the delta_x
            locals()["delta_x"+str(p)] = locals()["x_new"+str(p)] - locals()["x"+str(p)]
        for i in range(n):
            #Get the block agent i will send in interation k
            p = block_sending
            #Obtain the phi of block p
            locals()["phi_new"+str(p)] = new_phi#(locals()["phi"+str(p)])  
            #Get new x (adjacency_matrix is the correponding value in a)
            locals()["x_next"+str(p)] = adjacency_matrix * locals()["phi"+str(p)] * (locals()["x"+str(p)] + step_size_new*locals()["delta_x"+str(p)])/locals()["phi_new"+str(p)]
        for i in range(n):
            #Get the block agent i will send in interation k
            p = block_sending
            #Get the local gradient of x
            locals()["local_gradient"+str(p)] = local_grad_f#(locals()["x"+str(p)])
            #Get the local gradient of x_next
            locals()["local_gradient_new"+str(p)] = local_grad_f#(locals()["x_next"+str(p)])
            #Calculate the total gradient (adjacency_matrix is the correponding value in a)
            locals()["gradient_new"+str(p)] = adjacency_matrix * (locals()["phi_new"+str(p)]*locals()["gradient"+str(p)]+locals()["local_gradient_new"+str(p)]-locals()["local_gradient"+str(p)])/locals()["phi"+str(p)]
        # Save the result of objective value
        if max_iter==0 or (k % ceil(max_iter/epoch_size)) == 0:
            # Set x_now as the combination of all x1,x2,..,xb)
            x_now = ...#locals()["x"+str(i)] for i in range(b)       
            f_values[epoch_index] = f_obj#(x_now)
            epoch_index += 1
        # Based on calculations, update all the results
        for p in range(b):
            locals()["x"+str(p)] = locals()["x_next"+str(p)] + 0.0
            locals()["phi"+str(p)] = locals()["phi_new"+str(p)] + 0.0
            locals()["gradient"+str(p)] = locals()["gradient_new"+str(p)] + 0.0
        # Set x_next as the combination of all x1,x2,...,xb for returning
        x_next = ...#locals()["x"+str(i)] for i in range(b)
    return x_next, f_values

# Implementation for training the optimization model
n_attributes = len(dic_train_set[0].columns)-1
n_labels = len(dic_train_set[0])
mu_param = 10^-2
epoch_size = 5
max_iter = n_labels*epoch_size
step_size = 10**-6

t1_bs = time.time()    
x_sol_bs_1,f_vals_bs_1 = Block_sonata(dic_train_set,max_iter,epoch_size,x0,step_size,mu_param,b,n)
t2_bs = time.time()

x_sol_bs_2,f_vals_bs_2 = Block_sonata(dic_train_set,max_iter,epoch_size,x0,step_size,mu_param,b,n)

x_sol_bs_3,f_vals_bs_3 = Block_sonata(dic_train_set,max_iter,epoch_size,x0,step_size,mu_param,b,n)

print("Block_sonata:    f(x_",max_iter,sep='',end="") 
print(") = ","{:.3f}".format(f_vals_SGD_1[-1]), "     Time(second) = ", "{:.1f}".format(t2_bs-t1_bs))

fig = plt.figure(figsize=(8,6))

plt.plot(range(0,max_iter+1,ceil(max_iter/epoch_size)),f_vals_bs_1.tolist(),color='black',
         marker='v',markersize=10,linestyle='solid',label="SGD sample path 1",linewidth=4)

plt.plot(range(0,max_iter+1,ceil(max_iter/epoch_size)),f_vals_bs_2.tolist(),color='black',
         marker='v',markersize=10,linestyle='dashed',label="SGD sample path 2",linewidth=4)

plt.plot(range(0,max_iter+1,ceil(max_iter/epoch_size)),f_vals_bs_3.tolist(),color='black',
         marker='v',markersize=10,linestyle='dashdot',label="SGD sample path 3",linewidth=4)

plt.legend(loc=3,fontsize=12)
plt.xlabel('Number of single gradient evaluations', color='#1C2833',fontsize=12)
plt.ylabel("U(x)", color='#1C2833',fontsize=18)

plt.grid(True)

#Testing on the new data
def my_confu_mat(dic_test_set,opt_sol):
    npar_data= (dic_test_set[0]).to_numpy()
    test_data = npar_data[:,:-1]
    test_labels =npar_data[:,-1:]
    test_set_size = len(dic_test_set[0])
    pred_labels = np.zeros((test_set_size,1))
    for j in range(test_set_size):
        pred_labels[j][0] = np.sign(np.dot(test_data[j,:],opt_sol))
    output = confusion_matrix(test_labels, pred_labels)
    return output

def precision_score(dic_test_set,opt_sol):
    confu_mat = my_confu_mat(dic_test_set,opt_sol)
    output = confu_mat[1][1]/(confu_mat[1][0]+confu_mat[1][1])
    return output

ps_bs = precision_score(dic_test_set,x_sol_bs_1)

print("Block_sonata: ", "{0:.2%}".format(ps_bs))
my_confu_mat(dic_test_set,x_sol_bs_1)
















