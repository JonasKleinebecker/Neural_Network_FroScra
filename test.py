import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from neural_network_froscra import (
    Categorical_Crossentropy_Loss,
    Dense_Layer,
    MSE_Loss,
    Relu,
    Sigmoid,
    Softmax,
    accuracy,
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


class LinearRegression:
    def __init__(self):
        self.layer = Dense_Layer(1, 1)

    def forward(self, x):
        x = self.layer.forward(x, None)
        return x


def main():
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

    batch_size = 32
    for epoch in range(1000):
        print(f"Epoch {epoch}")

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        # Use tqdm to show progress for each batch
        for i in tqdm(range(0, len(X_train), batch_size)):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            y_pred = model.forward(X_batch)
            batch_accuracy = accuracy(y_batch, y_pred)
            batch_loss = loss.loss(y_batch, y_pred)

            total_loss += batch_loss
            total_accuracy += batch_accuracy
            num_batches += 1

            model.softmax.backward(loss.loss_prime(y_batch, y_pred))

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        print(f"Average Loss: {avg_loss}")
        print(f"Average Accuracy: {avg_accuracy}")


if __name__ == "__main__":
    main()
