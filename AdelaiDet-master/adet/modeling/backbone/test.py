import torch

A=torch.ones(4,3,1)    #2x3的张量（矩阵）
print("A:\n",A,"\nA.shape:\n", A.shape, "\n")

B=2*torch.ones(4,2,1)  #4x3的张量（矩阵）
print("B:\n",B,"\nB.shape:\n", B.shape, "\n")

C=torch.cat((A,B),1)  #按维数1（列）拼接
print("C:\n",C,"\nC.shape:\n", C.shape, "\n")