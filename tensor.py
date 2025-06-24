from __future__ import annotations
from typing import Union, Any
import numbers

import numpy as np

class Tensor:
    def __init__(self,
        arr: Union[np.ndarray, list, numbers.Number, Tensor],
        parents: list[Tensor] = [],
        requires_grad: bool = True,
        name: str = '',
        operation=None
    ):
        self._arr = arr
        self._parents = parents if parents is not None else []
        self.requires_grad = requires_grad
        self._name = name
        self._operation = operation
        self.grad = None

        self.__to_numpy()

        # Adjust parent/op info for direct input (not a result of an op)
        if not self._parents and not self._operation:
            self._name = f"in:{id(self) % 1000}"

    def __to_numpy(self):
        # Handle copying from another Tensor
        if isinstance(self._arr, Tensor):
            tensor = self._arr
            self._arr = tensor._arr.copy()
            self._parents = tensor._parents.copy()
            self.requires_grad = tensor.requires_grad
            self._operation = tensor._operation
            self.grad = tensor.grad.copy() if tensor.grad is not None else None
            self._name = tensor._name
            return self
        
        # Convert number or list to NumPy array
        if isinstance(self._arr, list):
            self._arr = np.array(self._arr, dtype=float)

        if isinstance(self._arr, numbers.Number):
            self._arr = np.array([self._arr], dtype=float)

        # Ensure it's a NumPy array
        if not isinstance(self._arr, np.ndarray):
            raise TypeError("Input must be a number, list, NumPy array or Tensor.")

        # Convert 1D arrays to column vectors
        if self._arr.ndim == 1:
            self._arr = self._arr.reshape(-1, 1)
        elif self._arr.ndim > 2:
            raise ValueError("Tensor must be 1D or 2D.")

        # Ensure dtype is float
        self._arr = self._arr.astype(float)

        return self

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self._arr))

    def numpy(self):
        return self._arr

    def __repr__(self):
        return f"Tensor({self._arr}, name={self._name}, shape={self._arr.shape})"
    
    def backward(self, my_grad=None):
        if not self.requires_grad:
            return

        if my_grad is None:
            # if self._arr.size != 1:
            #     raise RuntimeError("backward() só pode ser chamado sem argumentos em tensores escalares.")
            my_grad = Tensor(np.ones_like(self._arr))

        if self.grad is None:
            self.grad = my_grad
        else:
            self.grad = Tensor(self.grad.numpy() + my_grad.numpy(), requires_grad=False)

        if self._operation is not None:
            grads = self._operation.grad(my_grad, *self._parents)
            for parent, grad in zip(self._parents, grads):
                parent.backward(grad)
