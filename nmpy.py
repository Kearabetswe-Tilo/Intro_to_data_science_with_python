# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

a = np.arange(10, 30, 5) #like range but returns array
print(a)

a= np.arange(0, 2, 0.3) #takes float arguments
print(a)

a = np.arange(15)
print(a)

a = a.reshape(3, 5)
print(a)

print("shape:",  a.shape)
print("number of dimensions:",  a.ndim)
print("item type:", a.dtype.name)
print("item size:", a.itemsize)
print("array size:", a.size)
print("type:", type(a))


b = np.array([1, 2, 3])
print(b)
print("type b:", type(b))

c = np.array(([1, 2, 3], [4, 5, 6]), dtype = "complex")
print(c)
print("item type:", c.dtype.name)


d = np.zeros((3, 4))
print(d)
print("item type:", d.dtype.name)

d = np.ones((3, 4))
print(d)

d = np.empty((3, 4))
print(d)


x = np.arange(6) #1d array
print("x:\n", x)
y = np.arange(12).reshape(4, 3) #2d array
print("y:\n", y)
z = np.arange(24).reshape(2, 3, 4) #3d array
print("z:\n", z)

print(np.arange(10000))
print(np.arange(10000).reshape(100,100))
aa = np.arange(10000).reshape(100,100)


a = np.array([20,30,40,50])
b = np.arange(4)
c = a - b
print(c)
print(b ** 2)
print(10 * np.sin(a))
print(a < 35)


A = np.array( [[1,1], [0,1]] )
B = np.array( [[2,0], [3,4]] )
print(A * B) #elementwise product
print(A @ B) #matrix product
print(A.dot(B)) #also matrix product


from numpy import random as rg

a = np.ones((2,3), dtype=int)
b = rg.random((2,3))
a *= 3
print(a)
b += a
print(b)
#a += b #b is not automatically converted to integer type


from numpy import pi

a = np.ones(3, dtype = np.int32)
b = np.linspace(0, pi, 3)
print(a.dtype.name)
print(b.dtype.name)
c = a + b
print(c)
print(c.dtype.name)
d = np.exp(c * 1j)
print(d)
print(d.dtype.name)


from numpy import random as rg
a = rg.random((2, 3))
print(a)
print(a.sum())
print(a.min())
print(a.max())



b = np.arange(12).reshape(3, 4)
print(b)

print(b.sum(axis = 0)) #sum of each column
print(b.min(axis = 1)) #min of each row
print(b.cumsum(axis = 1)) #cumulative sum along each row


B = np.arange(3)
print(B)
print(np.exp(B))
print(np.sqrt(B))
C = np.array([2., -1., 4.])
print(np.add(B, C))


a = np.arange(10)**3
print(a)
print(a[2])
print(a[2:5])
#equivalent to a[0:6:2] = 1000;
#from start to position 6, exclusive, set every 2nd element to 1000
a[:6:2] = 1000
print(a)
print(a[ : :-1]) #reversed a
for i in a:
    print((int(i**(1/3))))
    

def f(x,y):
    return 10*x+y

b = np.fromfunction(f, (5, 4), dtype = int)
print(b)
print(b[2 , 3])
print(b[0:5, 1]) #each row in the second column of b
print(b[ : , 1]) #equivalent to the previous example
print(b[1:3, : ]) #each column in the second and third row of b
print(b[-1]) #the last row. Equivalent to b[-1,:]



c = np.array( [[[  0,  1,  2],   #a 3D array (two stacked 2D arrays)
                [ 10, 12, 13]],
               [[100, 101, 102],
               [110, 112, 113]]])
print(c.shape)

print(c[1,...]) #same as c[1,:,:] or c[1]
print(c[...,2]) #same as c[:,:,2]


def f(x,y):
    return 10*x+y

b = np.fromfunction(f, (5, 4), dtype = int)
for row in b:
    print(row)

for element in b.flat:
    print(element)

print(b)



from numpy import random as rg
a = np.floor(10 * rg.random((3,4)))
print(a)
print(a.shape)

print(a.ravel()) #returns the array, flattened
print(a.reshape(6,2)) #returns the array with a modified shape
print(a.T) #returns the array, transposed
print(a.T.shape)
print(a.shape)

b = np.floor(10 * rg.random((3,4)))
print(b)

print(np.vstack((a,b)))
print(np.hstack((a,b)))



from numpy import random as rg
a = np.floor(10*rg.random((2, 12)))
print(a)
#Split a into 3
print(np.hsplit(a, 3))

#Split a after the third and the fourth column
print(np.hsplit(a, (3, 4)))



a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)

print(a.transpose())

print(np.linalg.inv(a))

u = np.eye(2) #unit 2x2 matrix; "eye" represents "I"
print(u)

j = np.array([[0.0, -1.0], [1.0, 0.0]])

print(j @ j) #matrix product

print(np.trace(u))  #trace

a = np.array([[2 , 1], [1, -1]])
print(a)

b = np.array([[4], [-1]])
print(b)

x = np.linalg.inv(a).dot(b)
print(x)

a = np.array([[1, 2, 3], [4, 5, 2], [2, 8, 5]])
print(a)

b = np.array([5, 10, 15])
print(b)

x = np.linalg.inv(a).dot(b)
print(x)

y = np.linalg.solve(a, b)
print(y)



a = np.array([[1, 2, 3], [4, 5, 2], [2, 8, 5], [7, 9, 3]])
print(a)
print(a.sum())

print(a.sum(axis = 0)) #columnwise
print(a.sum(axis = 1)) #rowwise

print(a.mean())
print(a.mean(axis = 0)) #column means
print(a.mean(axis = 1)) 
print(np.median(a, axis = 1)) #row medians
print(np.std(a)) #variance









