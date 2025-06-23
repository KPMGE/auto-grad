from abc import ABC, abstractmethod

from tensor import Tensor

class Op(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tensor:
      pass

    @abstractmethod
    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
      pass

    @staticmethod
    def to_tensor(x) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return Tensor(x)
