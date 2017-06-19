"""Workflow for the machine learning part with Word2Vec."""
import time
from typing import List

from gensim.models import word2vec
import preprocessing


def preprocess_file(path, keep_stopwords=False, lemmatize=True):
    """Read the contents of a file and prepare it for w2v."""
    with open(path, 'r') as f:
        text = f.read()

    cleaned_text = preprocessing.prepare_for_w2v(text, keep_stopwords,
                                                 lemmatize)
    return cleaned_text


def train_model(sentences: List[List[str]], num_features=200,
                min_word_count=5, num_workers=4, context=5,
                downsampling=1e-3):
    """Launch the word2vec training on the sentences.

    Parameters
    ----------
    sentences: List[List[str]]
        list of sentences to train the model on

    num_features: int
        length of the word embedding vector for every word

    min_word_count: int
        threshold for word frequency: if a word appears less than this
        calue, it is discarded

    num_workers: int
        number of threads to use for training

    context: int
        width of the context taken into account to generate embeddings

    downsampling: float
        downsample value for frequent words (like stopwords)

    Returns
    -------
    model: model
        trained word2vec model
    """
    info = "Started training with {} features, {} minimum word count, ".format(
        num_features, min_word_count)
    info2 = "{} threads, {} context window, {} downsampling.".format(
        num_workers, context, downsampling)
    print(info + info2)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    #                     level=logging.INFO)

    start_time = time.time()
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)

    end_time = time.time()
    print("Finished training in {:.3f} seconds.".format(end_time - start_time))
    return model


def complete_workflow(save=True, keep_stopwords=False, lemmatize=True):
    """A convenience function to run all the workflow."""
    filepath = "train_data/Stephen Chbosky - Ragazzo da parete.txt"
    sentences = preprocess_file(filepath, keep_stopwords, lemmatize)

    features = 300
    word_count = 10
    workers = 4
    context = 10
    downsampling = 1e-3

    model = train_model(sentences, num_features=features,
                        min_word_count=word_count,
                        num_workers=workers,
                        context=context,
                        downsampling=downsampling)

    if save:
        save_model(model, features, word_count, context,
                   lemmatize, keep_stopwords)

    print("Workflow completed.")
    return model


def multiple_training():
    """Batch training function."""
    filepath = "train_data/Stephen Chbosky - Ragazzo da parete.txt"
    keep_stopwords = False
    lemmatize = True
    sentences = preprocess_file(filepath, keep_stopwords, lemmatize)

    features = 300
    word_count = [2, 5, 10]
    workers = 4
    context = [3, 5, 10]
    downsampling = 1e-3

    models = [[train_model(sentences, features, wc,
                           workers, ctx, downsampling)
               for wc in word_count]
              for ctx in context]

    return models, word_count, context


def save_model(model, num_features, min_word_count, context, lemmatized,
               keep_stopwords=False):
    """Save a word2vec model."""
    model_name = "saved_models/{}feat_{}minwords_{}context"\
                 .format(num_features, min_word_count, context)

    if lemmatized:
        model_name = model_name + "_lemmatized"

    if keep_stopwords:
        model_name = model_name + "_with_stopwords"
    else:
        model_name = model_name + "_no_stopwords"

    model.save(model_name)
    print("Saved model " + model_name)


def evaluate_model(model, n=50):
    """Evaluate the model with the target emotions.

    Parameters
    ----------
    model: gensim.Word2Vec
        the trained model to evaluate

    n: int
        the top n most similar words to ask for

    Returns
    -------
    goodness: float
        measure of good the model is
    """
    def measure_goodness(similar_words):
        """Give a goodness measure of similar_words."""
        fraction = n // 5
        neg = 0

        for i in range(len(similar_words)):
            word, similarity = similar_words[i]
            if i >= fraction and similarity >= 0.99:
                neg = neg + 1

        return neg

    emotions = ["rabbia", "anticipazione", "paura", "disgusto", "gioia",
                "tristezza", "fiducia", "sorpresa"]

    result = {}
    negatives = []
    for key in emotions:
        try:
            similar_words = model.most_similar(positive=[key], topn=n)
            negatives.append(measure_goodness(similar_words))
            result[key] = similar_words
        except KeyError:
            print("{} not in vocabulary".format(key))

    return result, negatives


if __name__ == "__main__":
    complete_workflow(save=True)
