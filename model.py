from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD


class KerasModel:

    def __init__(self, model_type="sequential", learning_rate=0.05, epochs=5, batch_size=1, **kwargs):
        # store inputted constants
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # allow for different model types depending on avaialble imports
        if model_type == "sequential":
            # could loop through user defined layers from command line args
            optimizer = SGD(learning_rate=learning_rate)
            self.model = Sequential()
            self.model.add(Dense(12, input_dim=kwargs["input_dim"], activation="relu"))
            # self.model.add(Dense(8, activation="relu"))
            # using softmax for the output layer to ensure values can be used as probabilities
            self.model.add(Dense(4, activation="softmax"))
            self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        else:
            raise Exception("Model type entered does not match acceptable models. Please enter an available model.")

    def train(self, X, y, score=True):
        # fit model based on inputted data and values for epochs and batch size
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

        if score:
            # calculate and print training score if needed
            _, accuracy = self.model.evaluate(X, y)
            print("Training Accuracy: %.2f" % (accuracy * 100))

    def test(self, X, y):
        # score model based on testing set
        _, accuracy = self.model.evaluate(X, y)
        print("Testing Accuracy: %.2f" % (accuracy * 100))

    def predict(self, X):
        # future method to predict on unlabeled data
        predictions = self.model.predict(X, batch_size=1)
        return predictions


if __name__ == "__main__":
    km = KerasModel()
