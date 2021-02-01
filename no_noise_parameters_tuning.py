#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 05:40:46 2020

@author: xiaoba
"""
import numpy as np
from numpy.random import default_rng
import time
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
def GN_slrp(Afun,X,tol=1e-04,maxit=200):
    k=X.shape[1]
    Ik=np.eye(k)
    XtX=X.T@X
    nrmX=np.trace(XtX)**0.5
    for i in range(1,maxit+1):
        Y = X@np.linalg.inv(XtX)
        Z = Afun(Y)   
        X =Z-X@((Y.T@Z-Ik)/2)
        XtX = X.T@X
        nrmXp=nrmX
        nrmX = np.trace(XtX)**0.5
        #check stopping
        crit=abs(1-nrmX/nrmXp)
        done=(crit<=tol)or (i==maxit)
        if done:
            break
    Xi=Y
    out={'Xi':Xi,'iter':i}
    return [X,out]
def objective(L, S, D,mu):
    
    return 0.5*np.linalg.norm(np.multiply(mask, L + S - D), 'fro')**2 + np.sum(np.abs(np.multiply(mask, S))) + mu*np.linalg.norm(np.linalg.svd(L, compute_uv=False), 1)
    

def prox_S(D,t):
   return np.multiply(mask, np.multiply(np.sign(D), np.maximum(np.abs(D) - t, 0)))
   

def prox_L_GN(D, t,mu):   # without large SVD
   global inneriter,M
   Afun=D@(D.T@M)
   def Afun(X):
       return D@(D.T@X)
   M,out    = GN_slrp(Afun, M,tol= 1e-8,maxit=1000 )
    
   inneriter   = inneriter + out['iter']
   U, S, V   = np.linalg.svd(M, full_matrices=False);
   S=np.diag(np.maximum(S - (t * mu), 0))
   L           = (U @ S @ V) @ (out['Xi'].T @ D)
   return L



rng = default_rng(100)
m=500
n=500
r_ratio=0.01
k=round(r_ratio*m)
A=rng.normal(size=(m, k)) @ rng.normal(size=(k, n))
c_ratio=0.2
J=rng.permutation(m * n)
J=J[0:round(c_ratio * m * n)]
y=A.reshape((250000,),order='F')
mean_y=np.mean(np.absolute(y))
noise=6*mean_y*rng.uniform(size=(round(c_ratio*m*n)))-3*mean_y
y[J]=noise



D=np.reshape(y,(m,n),order='F')

mask=rng.uniform(size=(m,n))<0.5
D=mask*D
p=k+25
error_lst=[]
i_lst=[]
GN_iter=[]

U, S, V = svds(D, p)
U[:, :p] = U[:, p-1::-1]
S = S[::-1]
S=np.diag(S)
V[:p, :] = V[p-1::-1, :]
M = U @ S

M=U@S


for alpha in np.arange(0.1,20,0.1):
    inneriter=0
    mu=18
   # timing=[]
    #for mu in np.arange(0.1,20,1):
    #increase alpha from 0.1 to greater while running
    t=1/alpha 
    tol=1e-4
    MAX_ITER=5000
    errL=np.zeros((MAX_ITER,1))
    #    fcnvalue=np.zeros((MAX_ITER,1))
    timerval=time.time()
    L=np.zeros((m,n))
    S=np.zeros((m,n))
    Z=np.zeros((m,n))
    for i in range(MAX_ITER):
        L_new= prox_L_GN(np.multiply(mask,D-S+t*Z)+np.multiply(1-mask,L),t,mu)
        S=prox_S(D-np.multiply(mask,L_new)+t*Z, t)
        Z=Z-alpha*(np.multiply(mask,L_new)+S-D)
        errL[i]=np.linalg.norm(L_new - L,'fro')
        print(str(i)+": "+str(errL[i]))
        #fcnvalue[i]=objective(L_new, S, D,mu)
        L=L_new
        print(inneriter)
        #iteration+=n_iter
        if errL[i]/np.linalg.norm(L,'fro')<tol:
            break
#        print(n_iter)
#    timing.append(time.time()-timerval)
    out={'errL':errL,'L':L,'S':S,'Z':Z,'i':i}
    fro_L        = np.linalg.norm(L - A, 'fro')/np.linalg.norm(A, 'fro')
    error_lst.append(fro_L)
    i_lst.append(i)
    GN_iter.append(inneriter)
print(error_lst)
print(i_lst)

fig, ax1 = plt.subplots()
ax1.set_xlabel('alpha')
ax1.set_ylabel('Relative error')
ax1.plot(np.arange(0.1,20,0.1),error_lst,label='Relative Error')
#ax1.set_ylim(0,0.0008)
ax1.tick_params(axis='y')
plt.legend()
#ax1.plot.ylim(bottom=0,top=0.0006)
ax2 = ax1.twinx() 
ax2.set_ylabel('iterations')
ax2.plot(np.arange(0.1,20,0.1),i_lst,color='gray',label='iterations')
ax2.tick_params(axis='y')
plt.grid()
fig.tight_layout()
plt.legend()