#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import default_rng
import time
from scipy.sparse.linalg import svds

'''
  Functions  
'''
def GN_slrp(Afun,D,X,**opts):
    if 'tol' not in opts:
        opts['tol']=1e-04
    if 'maxit' not in opts:
        opts['maxit']=200
    if 'quiet' not in opts:
        opts['quiet']=1
    if 'freq' not in opts:
        opts['freq']=1
    tol=opts['tol']
    maxit=opts['maxit']
    quiet=opts['quiet']
    freq=opts['freq']
    k=X.shape[1]
    Ik=np.eye(k)
    XtX=X.T@X
    nrmX=np.trace(XtX)
    for i in range(1,maxit+1):
        Y = X@np.linalg.inv(XtX)
        Z = Afun(D,Y)   
        X =Z-X@((Y.T@Z-Ik)/2)
        XtX = X.T@X
        nrmXp=nrmX
        nrmX = np.trace(XtX)
        #check stopping
        crit=abs(1-nrmX/nrmXp)
        done=(crit<=tol)or (i==maxit)
        if done:
            break
    Xi=Y
    out={'Xi':Xi,'iter':i}
    return [X,out]

def RPCA_S2Y(D,prox_S,prox_L,objective,t,**para):
    L = np.zeros(D.shape)
    tol=para['tol']
    errL=np.zeros((para['MAX_ITER'],1))
    fcnvalue=np.zeros((para['MAX_ITER'],1))
    timing=np.zeros((para['MAX_ITER'],1))
    S=prox_S(D-L)   #the gradient descent step
    timerval=time.time()
    for i in range(para['MAX_ITER']):
        L_new=prox_L(L-t*(S+L-D),t)
        S    = prox_S(D-L_new)    # the gradient descent step      
        errL[i]=np.linalg.norm(L_new - L,'fro')
        timing[i]=time.time()-timerval
        fcnvalue[i]=objective(L_new, S, D)
        timerval=time.time()
        L=L_new        
        if errL[i]/np.linalg.norm(L,'fro')<tol:
            break
    out={'errL':errL,'fcnvalue':fcnvalue,'time':timing,'i':i}
    return [L,S,out]

def objective(L, S, D):
    global mu,lam 
    val = 0.5*np.linalg.norm(L + S - D,'fro')**2 + lam * np.sum(np.absolute(S)) + mu * np.linalg.norm(np.linalg.svd(L,compute_uv=False))
    return val

def prox_S(D):
    global lam
    S = np.multiply(np.sign(D), np.maximum(np.absolute(D) - lam, 0))
    return S

def prox_L(D, t):  # with large SVD
    global mu,p 
    [U, S, V]   = np.linalg.svd(D)
    S=S[:p]
    U=U[:,:p]
    V=V[:p,:]
    S=np.maximum(np.diag(S) - (t * mu), 0)
    L=U@S@V 
    return L

def prox_L_GN(D, t):   # without large SVD
   global mu,M,inneriter
   Afun=D@(D.T@M)
   def Afun(X):
       return(D@(D.T@X))
   [M,out]    = GN_slrp(Afun,D, M,tol= 1e-8,maxit=1000 )
   inneriter   = inneriter + out['iter']
   [U, S, V]   = np.linalg.svd(M, full_matrices=False)
   S=np.maximum(np.diag(S) - (t * mu), 0)
   L           = (U @ S @ V) @ (out['Xi'].T @ D)
   print(inneriter)
   return L

'''
Main Algorithm
'''
#global lam,mu,p,M,inneriter

rng = default_rng(100)
m=500
n=500
r_ratio=0.05
k=round(r_ratio*m)
A=rng.normal(size=(m, k)) @ rng.normal(size=(k, n))
c_ratio=0.20
J=rng.permutation(m * n)
J=J[0:round(c_ratio * m * n)]
y=A.reshape((250000,),order='F')
mean_y=np.mean(np.absolute(y))
noise=6*mean_y*rng.uniform(size=(round(c_ratio*m*n)))-3*mean_y
y[J]=noise


#-----small moise----
small_noise=0.05*rng.normal(size=y.shape)
y=y+small_noise
D=np.reshape(y,(m,n),order='F')
p=k+5
t=1.7
U,S,V=svds(D,k=p,which='LM')
n = len(S)
# reverse the n first columns of u
U[:,:n] = U[:, n-1::-1]
# reverse s
S = S[::-1]
S=np.diag(S)
# reverse the n first rows of vt
V[:n, :] = V[n-1::-1, :]

M=U@S
inneriter=0
mu      = 0.
lam  = 0.02
t       = 1

[L, S,  out]    = RPCA_S2Y(D, prox_S, prox_L_GN,objective, t, MAX_ITER=5000,accelerated=1,tol=1e-04 )
fro_L        = np.linalg.norm(L - A, 'fro')/np.linalg.norm(A, 'fro')
i        = out['i']
