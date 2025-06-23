from functions import matmul
from tensor import Tensor

# z = Wx + b
W = Tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

b = Tensor([1.0, 2.0, 3.0])

z = matmul(W, b)
z.backward()

print(W.grad)
print(b.grad)
