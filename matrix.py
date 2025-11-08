'''import numpy as np
a1=np.array([[1,2,3],[4,5,6],[7,8,9]])
a2=np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Matrix1\n",a1)
print("Matrix2\n",a2)
print("sum",np.add(a1,a2))
print("diff",np.subtract(a1,a2))
print("pro",np.multiply(a1,a2))
print("Transpose A",a1.T)
print("Transpose A",a2.T)
print("Diagonal sum",sum(np.diag(a1)))
print("Diagonal sum",sum(np.diag(a2)))'''
import numpy as np
rows=int(input("enter elements"))
cols=int(input("enter elements"))
print("enter matrix1:")
data1=[]
for i in range(rows):
    row=list(map(int,input().split()))
    data1.append(row)
a1=np.array(data1)
print("Enter matrix 2:")
data2=[]
for i in range(rows):
    row=list(map(int,input().split()))
    data2.append(row)
a2=np.array(data2)
print("sum",np.add(a1,a2))
print("diff",np.subtract(a1,a2))
print("pro",np.multiply(a1,a2))
print("Transpose A",a1.T)
print("Transpose A",a2.T)
print("Diagonal sum",sum(np.diag(a1)))
print("Diagonal sum",sum(np.diag(a2)))

