import numpy as np

# -------------------------------------

# Problem 1

# initialize the matrices and vectors
v = np.array([9,5,10])
u = np.array([[4],[1],[3]])
A = np.array([[1,2,5],[3,4,6]])
B = np.array([[7,1,9],[2,2,3],[4,8,6]])
C = np.array([[8,6,5],[1,-3,4],[-1,-2,4]])

# u transpose * u
print(np.dot(u.T,u))
# u * u transpose
print(np.dot(u,u.T))
# v * u
print(np.dot(v,u))
# u + 5
print(np.add(u,5))
# A transpose
print(A.T)
# B ∗ u 
print(np.dot(B,u))
# B inverse
print(np.linalg.inv(B)) 
# B + C
print(np.add(B,C)) 
# B − C 
print(np.subtract(B,C))
# A ∗ B 
print(np.dot(A,B))
# B ∗ C
print(np.dot(B,C)) 
# B ∗ A
# will not work and raise error because B and A shapes are not compatible in this way
#print(np.dot(B,A))