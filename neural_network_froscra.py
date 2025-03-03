from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):
    """
    Abstract class for a module in a neural network.
    A module is a layer or an activation function.
    """

    @abstractmethod
    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        """
        Forward pass of the module.
        """
        pass

    @abstractmethod
    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the module.
        """
        pass

    @abstractmethod
    def backward(self, parent_input_grad: np.ndarray) -> None:
        """
        Backward pass of the module.
        """
        pass


class Relu(Module):
    """
    Rectified Linear Unit(ReLU) activation function class.
    """

    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        """
        Forward pass of the ReLU activation function.
        """
        self.parent = parent
        self.input = x
        return np.maximum(0, x)

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        """
        Calculate and return the input gradient of the ReLU activation function.
        """
        return np.where(self.input > 0, parent_input_grad, 0)

    def backward(self, parent_input_grad: np.ndarray) -> None:
        """
        Backward pass of the ReLU activation function.
        """
        if self.parent is None:
            return
        self.parent.backward(self.compute_input_grad(parent_input_grad))


class Sigmoid(Module):
    """
    Sigmoid activation function class.
    """

    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        """
        Forward pass of the Sigmoid activation function.
        """
        self.parent = parent
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        """
        Calculate and return the input gradient of the Sigmoid activation function.
        """
        return self.output * (1 - self.output) * parent_input_grad

    def backward(self, parent_input_grad: np.ndarray) -> None:
        """
        Backward pass of the Sigmoid activation function.
        """
        if self.parent is None:
            return
        self.parent.backward(self.compute_input_grad(parent_input_grad))


class Softmax(Module):
    """
    Softmax activation function class.
    """

    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        """
        Forward pass of the Softmax activation function.
        """
        self.parent = parent
        self.input = x
        self.output = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return self.output

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        """
        Calculate and return the input gradient of the Softmax activation function.
        """
        batch_size, _ = parent_input_grad.shape
        input_grad = np.zeros_like(parent_input_grad)

        for i in range(batch_size):
            softmax_output = self.output[i].reshape(-1, 1)  # Shape: (num_classes, 1)
            jacobian_matrix = np.diagflat(softmax_output) - np.dot(
                softmax_output, softmax_output.T
            )  # Shape: (num_classes, num_classes)
            input_grad[i] = np.dot(
                jacobian_matrix, parent_input_grad[i]
            )  # Shape: (num_classes,)

        return input_grad

    def backward(self, parent_input_grad: np.ndarray) -> None:
        """
        Backward pass of the Softmax activation function.
        """
        if self.parent is None:
            return
        self.parent.backward(self.compute_input_grad(parent_input_grad))


class Dense_Layer(Module):
    """
    Dense layer class.
    """

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
            self.weights = np.random.randn(input_size, output_size) * 0.01
        else:
            if weights.shape != (input_size, output_size):
                raise ValueError("Weights shape must be (input_size, output_size)")
            self.weights = weights
        if bias is None:
            self.bias = np.zeros((1, output_size))
        else:
            if bias.shape != (1, output_size):
                raise ValueError("Bias shape must be (1, output_size)")
            self.bias = bias

    def forward(self, x: np.ndarray, parent: Module | None) -> np.ndarray:
        """
        Forward pass of the dense layer.
        """
        self.output = x
        self.parent = parent
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def compute_weight_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        """
        Calculate and return the weight gradient of the dense layer.
        """
        return np.dot(self.input.T, parent_input_grad)

    def compute_bias_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        """
        Calculate and return the bias gradient of the dense layer.
        """
        return np.sum(parent_input_grad, axis=0, keepdims=True)

    def compute_input_grad(self, parent_input_grad: np.ndarray) -> np.ndarray:
        """
        Calculate and return the input gradient of the dense layer.
        """
        return np.dot(parent_input_grad, self.weights.T)

    def SGD_step(
        self,
        learning_rate: float,
        momentum: float,
        weights_grad: np.ndarray,
        bias_grad: np.ndarray,
    ) -> None:
        """
        Perform a Stochastic Gradient Descent (SGD) step for the dense layer.
        """
        self.velocity_weights = (
            momentum * self.velocity_weights - learning_rate * weights_grad
        )
        self.velocity_bias = momentum * self.velocity_bias - learning_rate * bias_grad
        self.weights += self.velocity_weights
        self.bias += self.velocity_bias

    def backward(self, parent_input_grad: np.ndarray) -> None:
        """
        Backward pass of the loss function for the dense layer.
        """
        if self.has_trainable_weights:
            self.SGD_step(
                learning_rate=0.01,
                momentum=0.9,
                weights_grad=self.compute_weight_grad(parent_input_grad),
                bias_grad=self.compute_bias_grad(parent_input_grad),
            )
        if self.parent is None:
            return
        self.parent.backward(self.compute_input_grad(parent_input_grad))


class Categorical_Crossentropy_Loss:
    """
    Categorical Crossentropy loss class.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the categorical crossentropy loss.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        eps = 1e-15
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the categorical crossentropy loss.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return y_pred - y_true


class MSE_Loss:
    """
    Mean Squared Error loss class.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the mean squared error loss.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return np.mean((y_true - y_pred) ** 2)

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the mean squared error loss.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return 2 * (y_pred - y_true)


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy for a given set of true and predicted labels.
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
