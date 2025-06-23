
from operation import Op
from tensor import Tensor
from name_manager import NameManager
import numpy as np
import math

class Add(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        b = self.to_tensor(args[1])
        result_arr = a.numpy() + b.numpy()

        return Tensor(
            result_arr,
            parents=[a, b],
            name=NameManager.new('add'),
            operation=self
        )

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
       return [
           Tensor(back_grad, name=NameManager.new('add_grad')),
           Tensor(back_grad, name=NameManager.new('add_grad'))
        ]

class Sub(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        b =  self.to_tensor(args[1])
        result_arr = a.numpy() - b.numpy()

        return Tensor(
            result_arr,
            parents=[a, b],
            requires_grad=a.requires_grad or b.requires_grad,
            name=NameManager.new('sub'),
            operation=self
        )

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a_grad = Tensor(back_grad, name=NameManager.new('sub_grad'))
        b_grad = Tensor(-1 * back_grad.numpy(), name=NameManager.new('sub_grad'))

        return [a_grad, b_grad]

class Prod(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        b = self.to_tensor(args[1])
        result_arr = a.numpy() * b.numpy()

        return Tensor(
            result_arr[0],
            parents=[a, b],
            name=NameManager.new('prod'),
            operation=self
        )

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        b = self.to_tensor(args[1])
        a_grad = back_grad.numpy() * b.numpy()
        b_grad = back_grad.numpy() * a.numpy()

        return [
            Tensor(a_grad, name=NameManager.new('prod_grad')),
            Tensor(b_grad, name=NameManager.new('prod_grad'))
        ]

class Sin(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        result_arr = np.sin(a.numpy())

        result = Tensor(
            result_arr,
            parents=[a],
            name=NameManager.new('sin'),
            operation=self
        )

        return result

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        a_grad = back_grad.numpy() * np.cos(a.numpy())

        return [Tensor(a_grad, name=NameManager.new('sin_grad'))]

class Cos(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        result_arr = np.cos(a.numpy())

        result = Tensor(
            result_arr,
            parents=[a],
            requires_grad=a.requires_grad,
            name=NameManager.new('cos'),
            operation=self
        )

        return result

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        a_grad = (-1) * back_grad.numpy() * np.sin(a.numpy())

        return [Tensor(a_grad, name=NameManager.new('cos_grad'))]

class Sum(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        result_value = np.sum(a.numpy())

        result = Tensor(
            arr=result_value,
            parents=[a],
            name=NameManager.new('sum'),
            operation=self
        )

        return result

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        grad_arr = np.ones_like(a.numpy()) * back_grad.numpy()

        return [Tensor(grad_arr, name=NameManager.new('sum_grad'))]
    
class Mean(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        tensor = self.to_tensor(args[0])
        mean = np.mean(tensor.numpy())

        return Tensor(
            arr=mean,
            parents=[tensor],
            name=NameManager.new('mean'),
            operation=self
        )

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        n = a.numpy().size
        grad_arr = 1/n * np.ones_like(a.numpy()) * back_grad.numpy()

        return [Tensor(grad_arr, name=NameManager.new('mean_grad'))]
    
class Square(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        result_arr = a.numpy() ** 2

        result = Tensor(
            result_arr,
            parents=[a],
            name=NameManager.new('square'),
            operation=self
        )

        return result

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        grad = 2 * a.numpy() * back_grad.numpy()

        return [Tensor(grad, name=NameManager.new('square_grad'))]

class MatMul(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        w = self.to_tensor(args[0])
        b = self.to_tensor(args[1])
        result_arr = w.numpy() @ b.numpy()
        
        return Tensor(
            result_arr,
            parents=[w, b],
            name=NameManager.new('matmul'),
            operation=self
        )

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        """Retorna a lista de derivadas parciais em relação aos pais (passados em args)"""
        w = self.to_tensor(args[0])
        b = self.to_tensor(args[1])
        w_grad = back_grad.numpy() @ b.numpy().T 
        b_grad = w.numpy().T @ back_grad.numpy()

        return [
            Tensor(w_grad, name=NameManager.new('matmul_grad')),
            Tensor(b_grad, name=NameManager.new('matmul_grad'))
        ]


class Exp(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        result_arr = np.exp(a.numpy())

        return Tensor(
            result_arr,
            parents=[a],
            name=NameManager.new('exp'),
            operation=self
        )


    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        grad = np.exp(a.numpy()) * back_grad.numpy()
        return [Tensor(grad, name=NameManager.new('exp_grad'))]

class ReLU(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        relu = np.vectorize(self._relu_fn)
        result_arr = relu(a.numpy())

        return Tensor(
            result_arr,
            parents=[a],
            name=NameManager.new('relu'),
            operation=self
        )

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        relu_grad = np.vectorize(self._relu_grad)
        grad = relu_grad(a.numpy()) * back_grad.numpy()

        return [Tensor(grad, name=NameManager.new('relu_grad'))]

    def _relu_fn(self, x: float) -> float:
        return x if x > 0 else 0
    
    def _relu_grad(self, x: float) -> float:
        return 1 if x > 0 else 0

class Sigmoid(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        sigmoid = np.vectorize(self._sigmoid)
        result = sigmoid(a.numpy())

        return Tensor(
            result,
            parents=[a],
            name=NameManager.new('sigmoid'),
            operation=self
        ) 

    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        sigmoid = np.vectorize(self._sigmoid)
        sigmoid_result = sigmoid(a.numpy())
        grad = sigmoid_result * (1 - sigmoid_result)

        return [Tensor(grad * back_grad.numpy(), name=NameManager.new('sigmod_grad'))]

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

class Tanh(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        result_value = np.tanh(a.numpy())

        return Tensor(
            arr=result_value,
            parents=[a],
            name=NameManager.new('tanh'),
            operation=self
        )


    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        a = self.to_tensor(args[0])
        grad = (1 - np.tanh(a.numpy()) ** 2) * back_grad.numpy()

        return [Tensor(grad, name=NameManager.new('tanh_grad'))]

class Softmax(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        a = self.to_tensor(args[0])
        result = self._softmax(a.numpy())

        return Tensor(
            arr=result,
            parents=[a],
            name=NameManager.new('softmax'),
            operation=self
        )


    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        x_tensor = self.to_tensor(args[0])
        x = x_tensor.numpy().flatten()
        y = self._softmax(x_tensor.numpy()).flatten()
        j = self._jacobian(x, y)

        return [Tensor(j.T @ back_grad.numpy(), name=NameManager.new('softmax_grad'))]
    
    def _softmax(self, a: np.ndarray) -> np.ndarray:
        a_exp = np.exp(a)
        a_sum = np.sum(a_exp)
        return a_exp / a_sum
    
    def _jacobian(self, x: list, y: list) -> np.ndarray:
        jacobian = np.zeros((len(y), len(y)))

        for (k, _) in enumerate(x):
            for (i, _) in enumerate(y):
                if i == k:
                    jacobian[i][k] = y[i] * (1 - y[i])
                else:
                    jacobian[i][k] = -(y[k] * y[i])

        return jacobian


softmax = Softmax()
tanh = Tanh()
sigmoid = Sigmoid()
relu = ReLU()
exp = Exp()
matmul = MatMul()
square = Square()
mean = Mean()
my_sum = Sum()
cos = Cos()
sin = Sin()
prod = Prod()
sub = Sub()
add = Add()