import pandas as pd
import string
import sys
import sklearn
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


class DataProcess:
    """
    class to process train and test dataset for ml model

    INPUTS:
        ngram_min (int): minimum number of words to use a n grams in word vectorizer
        ngram_max (int): maximum number of words to use a n grams in word vectorizer
    """

    def __init__(self, ngram_min=1, ngram_max=1, svd_components=10):
        # set article classes based on assigned int
        self.text_class = {0: "World", 1: "Sports", 2: "Business", 3: "Sci / Tech"}

        # allow adjustable ngram inputs
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

        # build word vectorizer based on inputted ngram values and below tokenizer
        self.count_vec = CountVectorizer(stop_words="english", tokenizer=self.tokenize,
                                         ngram_range=(ngram_min, ngram_max))

        # svd to reduce matrix down to 10 components
        self.svd = TruncatedSVD(n_components=svd_components)

    def tokenize(self, text):
        # static method but allowing it to be associated with processing class

        # remove all punctuation from text
        text = "".join([ch for ch in text if ch not in string.punctuation])

        # tokenize and stem each word to account for plural, tense, etc.
        tokens = word_tokenize(text)
        stems = []
        for item in tokens:
            # stemming using PorterStemmer
            # could easily change this to allow for stemming selection from fixed NLTK list
            stems.append(PorterStemmer().stem(item))

        # rejoin string to be entered into dataframe
        return " ".join(stems)

    def process(self, data, train=False):
        # set text to lowercase to ensure words are compared regardless of case
        data["content"] = data["content"].str.lower()

        if train:
            # if set to training, fit and transform count vectorizer
            # vectorizer should not be refit for training data
            word_counts = self.count_vec.fit_transform(data["content"])
            word_counts_reduced = self.svd.fit_transform(word_counts)
        else:
            try:
                word_counts = self.count_vec.transform(data["content"])
                word_counts_reduced = self.svd.fit_transform(word_counts)
            except sklearn.exceptions.NotFittedError:
                # raise exception if attempting to test model without previously fitting to training data
                raise Exception("Vectorizer not yet fitted, please repeat this process with training data "
                                "and set train=True.")
            except:
                print(sys.exec_info()[0])
                raise Exception("Uncaught error attempting to transform inputted data.")

        return word_counts_reduced

    def class_encoder(self, data_classes):
        # create dummy variable for prediction classes
        dummies = pd.get_dummies(data_classes)

        # ability to reconstruct dummies into categorical variables
        x = dummies.stack()
        reconstructed = pd.Series(pd.Categorical(x[x != 0].index.get_level_values(1)))

        return dummies


if __name__ == "__main__":
    # main block used for class testing/development
    data_processor = DataProcess(ngram_min=1, ngram_max=1)

    train_df = pd.read_csv("data/train.csv", names=["class", "title", "content"], header=None)

    data_processor.process(csv_filepath="data/train.csv", train=True)

    test_df = pd.read_csv("data/test.csv", names=["class", "title", "content"], header=None)
