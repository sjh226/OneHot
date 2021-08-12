import pandas as pd
import sys
from process_data import DataProcess
from model import KerasModel


def main(ngram_min, ngram_max, svd_components, learning_rate, epochs, batch_size):
    # pull training and testing data from local csv
    train_data = pd.read_csv("data/train.csv", names=["class", "title", "content"], header=None)
    test_data = pd.read_csv("data/test.csv", names=["class", "title", "content"], header=None)

    # shortened data set for development purposes
    # train_data = train_data.iloc[0:20, :]

    # create data process class for feature extraction
    data_processor = DataProcess(ngram_min=ngram_min, ngram_max=ngram_max, svd_components=svd_components)
    training_vectors = data_processor.process(data=train_data, train=True)
    testing_vectors = data_processor.process(data=test_data, train=False)

    # pull prediction classes from training/testing data and create dummy variables for prediction
    training_classes = train_data["class"]
    testing_classes = test_data["class"]
    training_dummies = data_processor.class_encoder(training_classes)
    testing_dummies = data_processor.class_encoder(testing_classes)

    # build classification model (sequential) based on training data
    # dynamically find training dimension from inputted svd value
    km = KerasModel(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                    input_dim=training_vectors.shape[1])
    # allow for additional training epochs in more iterations
    km.train(X=training_vectors, y=training_dummies)
    km.test(X=testing_vectors, y=testing_dummies)


if __name__ == "__main__":
    # currently just runs the model on train/test data
    # TODO: allow for saving of trained model and inputted prediction data through command line arguments

    # hyperparameters can be set through command line arguments
    # TODO: argparser allowing for specific parameters to be set
    arg_dic = {}
    for arg_idx in range(1, len(sys.argv)):
        arg = sys.argv[arg_idx]
        arg_value_pair = arg.split("=")
        arg_dic[arg_value_pair[0]] = arg_value_pair[1]

    # very clunky way to get values from command line inputs, better to use arg parse
    if "ngram_min" in arg_dic:
        ngram_min = arg_dic["ngram_min"]
    else:
        ngram_min = 1
    if "ngram_max" in arg_dic:
        ngram_max = arg_dic["ngram_max"]
    else:
        ngram_max = 1
    if "svd_components" in arg_dic:
        svd_components = arg_dic["svd_components"]
    else:
        svd_components = 10
    if "learning_rate" in arg_dic:
        learning_rate = arg_dic["learning_rate"]
    else:
        learning_rate = 0.05
    if "epochs" in arg_dic:
        epochs = arg_dic["epochs"]
    else:
        epochs = 5
    if "batch_size" in arg_dic:
        batch_size = arg_dic["batch_size"]
    else:
        batch_size = 1

    main(ngram_min, ngram_max, svd_components, learning_rate, epochs, batch_size)
