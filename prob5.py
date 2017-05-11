# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:28:38 2017

@author: whisp
"""
# Prob 4
import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(10,250,100000)
y = np.zeros(100000)
alpha = 60
beta = 120

for i in range(len(r)):
    r_ = r[i]
    y[i] = max(((r_-alpha)/(r_+alpha))**2, ((r_-beta)/(r_+beta))**2)
    
plt.plot(r,y)
print(np.sqrt(alpha*beta))
print(r[np.argmin(y)])
print(np.sqrt(alpha*beta)-r[np.argmin(y)])

#%%
import numpy as np
import matplotlib.pyplot as plt

steps = 100000
error1 = np.ndarray([steps])
n = 10
b = np.ones(n)
x = np.random.randn(n)
x0 = x
A = np.ndarray([n, n])
for i in range(n):
    for j in range(n):
        A[i, j] = 1/(i + j +1)

# True answer
ans = np.linalg.solve(A, b) 
plt.figure()       

# Single-Step method=SSM=Gauss-Seidel method
for step in range(steps):
    for j in range(n):
        sum1 = 0
        for i in range(n):
            if i != j:
                sum1 += A[j, i] * x[i]
        x[j] = (-sum1 + b[j])/A[j, j]
    error1[step] = max(ans - x)       
print(ans - x)
plt.plot(range(steps),error1, label="Gauss-Seidel")


# Total-Step method=TSM=Jacobi method

x = x0 # Start from the same initial point
error2 = np.ndarray([steps])
for step in range(steps):
    x_old = x
    for j in range(n):
        sum1 = 0
        sum2 = 0
        for i in range(n):
            if i < j:
                sum1 += A[j, i] * x[i]
            if i > j:
                sum2 += A[j, i] * x_old[i]
        x[j] = (- sum1 - sum2 + b[j])/A[j, j]
    error2[step] = max(ans - x) 
print(ans - x)
plt.plot(range(steps),error2, label="Jacobi")


# Relaxation Methods
omega = 1.2
x = x0 # Start from the same initial point
error3 = np.ndarray([steps])
for step in range(steps):
    x_old = x
    for j in range(n):
        sum1 = 0
        for i in range(n):
            if i != j:
                sum1 += A[j, i] * x[i]
        x[j] = x_old[j] + omega * ((-sum1 + b[j])/A[j, j] - x_old[j])
    error3[step] = max(ans - x)       
print(ans - x)

plt.xlabel('steps')
plt.ylabel('error')
plt.title('Comparasion of convergence n=%d' % n)
plt.plot(range(steps),error3, label='Relaxation')
plt.legend()
plt.savefig('n%d-all.pdf' % n)

plt.figure()
plt.xlabel('steps')
plt.ylabel('error')
plt.subplot(211)
plt.title('Comparasion of convergence n=%d' % n)
plt.plot(range(steps),error1, 'C0', label="Gauss-Seidel") 
plt.plot(range(steps),error2, 'C1', label="Jacobi")
plt.legend()

plt.subplot(212)
plt.plot(range(steps),error2, 'C1', label="Jacobi") 
plt.plot(range(steps),error3, 'C2', label='Relaxation')
plt.legend()
plt.savefig('n%d.pdf' % n)


# Relaxation Methods
plt.figure()
for omega in np.asarray([1.1, 1.2, 1.8]):
    x = x0 # Start from the same initial point
    error3 = np.ndarray([steps])
    for step in range(steps):
        x_old = x
        for j in range(n):
            sum1 = 0
            for i in range(n):
                if i != j:
                    sum1 += A[j, i] * x[i]
            x[j] = x_old[j] + omega * ((-sum1 + b[j])/A[j, j] - x_old[j])
        error3[step] = max(ans - x)       
    print(ans - x)
    plt.plot(range(steps),error3, label = 'omega = %2.1f' % omega)
plt.title('Comparasion of omega n=%d' % n)
plt.xlabel('steps')
plt.ylabel('error')
plt.legend()
plt.savefig('omega n%d.pdf' % n)


