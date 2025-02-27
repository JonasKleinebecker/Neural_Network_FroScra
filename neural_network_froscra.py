from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        pass

    @abstractmethod
    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def loss_backwards(self, parent_input_grad: np.ndarray) -> None:
        pass


class Relu(Module):
    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        self.parent = parent
        self.meaned_input = np.mean(x, axis=0)
        self.meaned_input = self.meaned_input.reshape(1, -1)
        return np.maximum(0, x)

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        return np.where(self.meaned_input > 0, parent_input_grad, 0)

    def loss_backwards(self, parent_input_grad: np.ndarray) -> None:
        self.parent.loss_backwards(self.compute_input_grad(parent_input_grad))


class Sigmoid(Module):
    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        self.parent = parent
        self.input = x
        output = 1 / (1 + np.exp(-x))
        self.meaned_output = np.mean(output, axis=0)
        self.meaned_output = self.meaned_output.reshape(1, -1)
        return output

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        return self.meaned_output * (1 - self.meaned_output) * parent_input_grad

    def loss_backwards(self, parent_input_grad: np.ndarray) -> None:
        self.parent.loss_backwards(self.compute_input_grad(parent_input_grad))


class Softmax(Module):
    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        self.parent = parent
        self.input = x
        output = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.meaned_output = np.mean(output, axis=0)
        self.meaned_output = self.meaned_output.reshape(1, -1)
        return output

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        s = self.meaned_output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
        return np.dot(jacobian_matrix, parent_input_grad.T).T

    def loss_backwards(self, parent_input_grad: np.ndarray) -> None:
        self.parent.loss_backwards(self.compute_input_grad(parent_input_grad))


class Dense_Layer(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_trainable_weights: bool = True,
        weights: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ):
        self.velocity_weights = np.zeros((input_size, output_size))
        self.velocity_bias = np.zeros((1, output_size))
        self.has_trainable_weights = has_trainable_weights
        if weights is None:
            self.weights = np.random.randn(input_size, output_size)
        else:
            if weights.shape != (input_size, output_size):
                raise ValueError("Weights shape must be (input_size, output_size)")
            self.weights = weights
        if bias is None:
            self.bias = np.random.randn(1, output_size)
        else:
            if bias.shape != (1, output_size):
                raise ValueError("Bias shape must be (1, output_size)")
            self.bias = bias

    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        self.parent = parent
        self.input = np.mean(x, axis=0)
        self.input = self.input.reshape(1, -1)
        return np.dot(x, self.weights) + self.bias

    def compute_weight_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        return np.dot(self.input.T, parent_input_grad)

    def compute_bias_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        return np.sum(parent_input_grad, axis=0)

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        return np.dot(parent_input_grad, self.weights.T)

    def SGD_step(
        self,
        learning_rate: float,
        momentum: float,
        weights_grad: np.ndarray,
        bias_grad: np.ndarray,
    ) -> None:
        self.velocity_weights = (
            momentum * self.velocity_weights - learning_rate * weights_grad
        )
        self.velocity_bias = momentum * self.velocity_bias - learning_rate * bias_grad
        self.weights += self.velocity_weights
        self.bias += self.velocity_bias

    def loss_backwards(self, parent_input_grad: np.ndarray) -> None:
        if self.has_trainable_weights:
            self.SGD_step(
                learning_rate=0.01,
                momentum=0.9,
                weights_grad=self.compute_weight_grad(parent_input_grad),
                bias_grad=self.compute_bias_grad(parent_input_grad),
            )
        if self.parent is None:
            return
        self.parent.loss_backwards(self.compute_input_grad(parent_input_grad))


class Categorical_Crossentropy_Loss:
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        loss_prime = np.mean(y_pred - y_true, axis=0)
        return np.mean(y_pred - y_true, axis=0).reshape(1, -1)


class MSE_Loss:
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return np.mean((y_true - y_pred) ** 2, axis=1)

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return np.mean(2 * (y_pred - y_true), axis=0)
