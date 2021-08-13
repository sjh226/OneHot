# OneHot

This module is designed to read a previously downloaded csv of training data containing brief contents of a news article and their associated genre (World, Sports, Business, and Sci/Tech). The data is used to train a simple neural network based off of the Keras model API. The contents of each article are broken down into key components by eliminating punctuation and counting the stemmed string of each word (or groupings of words depending on user input). After a final dimensionality reduction, the training data is sent through the Keras model which is built on 2 layers in a sequential model structure.

Once trained, the model can be tested and scored on a downloaded csv containing the test data.

Python module version requirements are listed in the requirements.txt file.

The module can bet executed using the default hyperparameter inputs from the command line in the following fashion:

    python main.py

Additional parameters may be set by the user by changing the values in the code block below.

    python main.py ngram_min=1 ngram_max=1 svd_components=10 learning_rate=0.05 epochs=5 batch_size=1
    
Each parameter specified above can be set individually or as a group of multiple members.
