#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 10:59:35 2019

@author: jun
"""
import numpy as np
from numpy.linalg import norm
pi=np.pi
sin = np.sin
cos=np.cos


def heat_2d(N,L):
    h=1/N
    delta_t=0.0001
# In[]:系数矩阵 A 和边界处理
    x=np.linspace(0,1,N+1)
    y=np.linspace(0,1,N+1)
    X,Y=np.meshgrid(x,y)
    B=np.zeros(shape=(N+1,N+1),dtype=float)
    C=np.zeros(shape=(N+1,N+1),dtype=float)
    U=np.zeros(shape=((N+1)**2,1),dtype=float)
    F=np.zeros(shape=((N+1)**2,1),dtype=float)
        
    for i in range(1,N):
        C[i,i]=delta_t/h**2
        B[i,i-1]=delta_t/h**2
        B[i,i]=1-4*delta_t/h**2
        B[i,i+1]=delta_t/h**2
    A=np.kron(np.eye(N+1),B)+np.kron(np.eye(N+1,k=1)+np.eye(N+1,k=-1),C)
    #边界处理
    for i in range(N+1):
        for j in range(N+1):
            if i==j:
                A[i,i]=1
                A[N**2+N+i,N**2+N+j]=1
            else:
                A[i,j]=0
                A[N**2+N+i,N**2+N+j]=0
    
    for i in range(N-1):
        A[(i+1)*(N+1),(i+1)*(N+1)]=1
        A[N+(i+1)*(N+1),N+(i+1)*(N+1)]=1
    for i in range(N+1):
        for j in range(N+1):
            A[i,N+j+1]=0
            A[N**2+N+i,N**2+j-1]=0
# In[]:对时间循环
    for t in range(1+int(L/delta_t)):
        U = A@U +delta_t*F
        F=(sin(5*pi*delta_t*t)*sin(2*pi*X)*sin(pi*Y)).reshape(-1,1)
        for i in range(N+1):
            F[i]=0
            F[N**2+N+i]=0
        for i in range(N-1):
             F[(i+1)*N+1]=0
             F[N+(i+1)*(N+1)]=0
    return X,Y,U

# In[]: 求误差
def error(N,U,time):
    X,Y,U_exact=heat_2d(40,time)
    U_new_exact=[]
    if N==10:
        for i in range(N+1):
            for j in range(N+1):
                U_new_exact.append(U_exact[41*4*i+4*j])
                #U_new_exact = np.array(U_new_exact)
    elif N==20:
         for i in range(N+1):
            for j in range(N+1):
                U_new_exact.append(U_exact[41*2*i+2*j])
                #U_new_exact = np.array(U_new_exact)
    error = norm(U_new_exact-U,ord=2)/((N+1)**2)
    return error
# In[]:数值结果
N1=10
L1=0.1
X1,Y1,U1=heat_2d(N1,L1)
e1 = error(N1,U1,L1)
print('N=%d,l2误差为%.6f'%(N1,e1))
# In[]:
N2=10*2
X2,Y2,U2=heat_2d(N2,L1)
e2 = error(N2,U2,L1)
R1=np.math.log(e1/e2)/np.math.log(2)
print('N=%d,l2误差为%.6f,误差阶为%.6f'%(N2,e2,R1))
# In[]
# In[]
N3=10
L2=0.2
X3,Y3,U3=heat_2d(N3,L2)
e3 = error(N3,U3,L2)
print('N=%d,l2误差为%.6f'%(N3,e3))
# In[]:
N4=10*2
L2=0.2
X4,Y4,U4=heat_2d(N4,L2)
e4 = error(N4,U4,L2)
R2=np.math.log(e3/e4)/np.math.log(2)
print('N=%d,l2误差为%.6f,误差阶为%.6f'%(N4,e4,R2))
# In[]:
# In[]
N5=10
L3=0.4
X5,Y5,U5=heat_2d(N5,L3)
e5 = error(N5,U5,L3)
print('N=%d,l2误差为%.6f'%(N5,e5))
# In[]:
N6=10*2
X6,Y6,U6=heat_2d(N6,L3)
e6 = error(N6,U6,L3)
R3 = np.math.log(e5/e6)/np.math.log(2)
print('N=%d,l2误差为%.6f,误差阶为%.6f'%(N6,e6,R3))
# In[]:
# In[]
N7=10
L4=0.8
X7,Y7,U7=heat_2d(N7,L4)
e7 = error(N7,U7,L4)
print('N=%d,l2误差为%.6f'%(N7,e7))
# In[]:
N8=10*2
X8,Y8,U8=heat_2d(N8,L4)
e8 = error(N8,U8,L4)
R4 = np.math.log(e7/e8)/np.math.log(2)
print('N=%d,l2误差为%.6f,误差阶为%.6f'%(N8,e8,R4))
# In[]:Draw picture

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(1)
ax=Axes3D(fig)
X1,Y1,uh1=heat_2d(N2,L1)
uh1=np.reshape(uh1,(N2+1,N2+1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X1,Y1,uh1,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

fig=plt.figure(2)
ax=Axes3D(fig)
X2,Y2,uh2=heat_2d(N4,L2)
uh2=np.reshape(uh2,(N4+1,N4+1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X2,Y2,uh2,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

fig=plt.figure(3)
ax=Axes3D(fig)
X3,Y3,uh3=heat_2d(N6,L3)
uh3=np.reshape(uh3,(N6+1,N6+1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X3,Y3,uh3,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

fig=plt.figure(4)
ax=Axes3D(fig)
X4,Y4,uh4=heat_2d(N8,L4)
uh4=np.reshape(uh4,(N8+1,N8+1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X4,Y4,uh4,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

plt.show()

    
        
