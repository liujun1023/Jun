import numpy as np
from numpy.linalg import norm,inv
sin=np.sin
cos=np.cos
pi=np.pi
# In[]:
def para(N):
    T=1
    tau=0.01
    h=1/N
    x=np.linspace(0,1,N,endpoint=False)
    x = x[1:]
    y=np.linspace(0,1,N,endpoint=False)
    y = y[1:]
    #初始值
    X,Y = np.meshgrid(x,y)
    U0=sin(pi*X)*cos(pi*Y)
    Un = np.reshape(U0,(-1,1))
    
    # In[]:系数矩阵
    
    a = 2+tau/(8*(h**2))
    b = -tau/(16*(h**2))
    c = 2-tau/(8*(h**2))
    d = tau/(16*(h**2))
    B1 = a*np.eye(N-1)+b*np.eye(N-1,k=1)+b*np.eye(N-1,k=-1)
    B2 = c*np.eye(N-1)+d*np.eye(N-1,k=1)+d*np.eye(N-1,k=-1)
    
    A1 = np.kron(np.eye(N-1),B1)
    
    A2 = np.kron(np.eye(N-1),c*np.eye(N-1))+np.kron(np.eye(N-1,k=1)+np.eye(N-1,k=-1),d*np.eye(N-1))
    A3 = np.kron(np.eye(N-1),a*np.eye(N-1))+np.kron(np.eye(N-1,k=1)+np.eye(N-1,k=-1),b*np.eye(N-1))
    A4 = np.kron(np.eye(N-1),B2)
    for i in range(N-1):
        A2[i,i]=c+d
        A3[i,i]=a+b
    for j in range(N**2-3*N+2,(N-1)**2):
        A2[j,j]=c+d
        A3[j,j]=a+b
        
    # In[]:解方程 
    for t in range(int(T/tau)):
        U1 = inv(A1)@A2@Un
        Un = inv(A3)@A4@U1
        exact_sol=sin(pi*X)*cos(pi*Y)*np.exp((-pi**2*(t+1)*tau)/8)
    return X,Y,Un.reshape(N-1,N-1),exact_sol
# In[]:误差
N1=4
X1,Y1,uh1,u_true1=para(N1)
e1=norm((uh1-u_true1).reshape(-1,1),ord=2)/((N1-1)**2)
print('N=%d,l2误差为%.6f'%(N1,e1))
print('数值解为',uh1)
print('解析解为',u_true1)

N2=8
X2,Y2,uh2,u_true2=para(N2)
e2=norm((uh2-u_true2).reshape(-1,1),ord=2)/((N2-1)**2)
R1=np.math.log(e1/e2)/np.math.log(2)
print('N=%d,l2误差为%.6f,误差阶为%.6f'%(N2,e2,R1))

N3=16
X3,Y3,uh3,u_true3=para(N3)
e3=norm((uh3-u_true3).reshape(-1,1),ord=2)/((N3-1)**2)
R2=np.math.log(e2/e3)/np.math.log(2)
print('N=%d,l2误差为%.6f,误差阶为%.6f'%(N3,e3,R2))

N4=32  
X4,Y4,uh4,u_true4=para(N4)    
e4=norm((uh4-u_true4).reshape(-1,1),ord=2)/((N4-1)**2)    
R3=np.math.log(e3/e4)/np.math.log(2)                                
print('N=%d,l2误差为%.6f,误差阶为%.6f'%(N4,e4,R3))
# In[]:Draw picture

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(1)
ax=Axes3D(fig)
X1,Y1,uh1,exact_sol=para(N1)
uh1=np.reshape(uh1,(N1-1,N1-1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X1,Y1,uh1,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

fig=plt.figure(2)
ax=Axes3D(fig)
X2,Y2,uh2,exact_sol=para(N2)
uh2=np.reshape(uh2,(N2-1,N2-1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X2,Y2,uh2,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

fig=plt.figure(3)
ax=Axes3D(fig)
X3,Y3,uh3,exact_sol=para(N3)
uh3=np.reshape(uh3,(N3-1,N3-1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X3,Y3,uh3,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

fig=plt.figure(4)
ax=Axes3D(fig)
X4,Y4,uh4,exact_sol=para(N4)
uh4=np.reshape(uh4,(N4-1,N4-1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x,y)')
ax.plot_surface(X4,Y4,uh4,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
plt.title('uh')

plt.show()