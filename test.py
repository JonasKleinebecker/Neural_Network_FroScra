import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network_froscra import (
    Categorical_Crossentropy_Loss,
    Dense_Layer,
    MSE_Loss,
    Relu,
    Sigmoid,
    Softmax,
)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer1 = Dense_Layer(self.input_size, self.hidden_size)
        self.layer2 = Dense_Layer(self.hidden_size, self.output_size)
        self.relu = Relu()
        self.softmax = Softmax()

    def forward(self, x):
        x = self.layer1.forward(x, None)
        x = self.relu.forward(x, self.layer1)
        x = self.layer2.forward(x, self.relu)
        x = self.softmax.forward(x, self.layer2)
        return x


def main():
    # prep data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))

    y_train = np.eye(output_size)[y_train]
    y_test = np.eye(output_size)[y_test]

    print(input_size, output_size)
    print(X_train.shape, y_train.shape)
    print(X_train[:5], y_train[:5])

    model = NeuralNetwork(input_size, 32, output_size)
    loss = Categorical_Crossentropy_Loss()

    # train model
    for epoch in range(1000):
        print(f"Epoch {epoch}")
        y_pred = model.forward(X_train)
        loss_value = loss.loss(y_train, y_pred)
        print(f"Loss: {loss_value}")
        model.softmax.loss_backwards(loss.loss_prime(y_train, y_pred))


if __name__ == "__main__":
    main()
