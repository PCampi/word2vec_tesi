"""Workflow for the machine learning part with Word2Vec."""
import glob
import logging
import time
from gensim.models import word2vec

import preprocessing


def preprocess_file(path, keep_stopwords=False, lemmatize=True):
    """Read the contents of a file and prepare it for w2v."""
    with open(path, 'r') as f:
        text = f.read()

    cleaned_text = preprocessing.prepare_for_w2v(text, keep_stopwords,
                                                 lemmatize)
    return cleaned_text


def get_corpus(train_dir="./train_data"):
    """Get all the text as a single string."""
    training_files = glob.glob(train_dir + '/*.txt')
    book_count = len(training_files)

    corpus = ""
    for f in training_files:
        with open(f, 'r') as new_file:
            corpus = corpus + new_file.read()

    return corpus, book_count


def complete_workflow(save=True, keep_stopwords=False, lemmatize=True,
                      train_dir="./train_data"):
    """A convenience function to run all the workflow."""
    # 1. get the text to analyze
    corpus, book_count = get_corpus(train_dir)

    # 2. set the model parameters
    features = 200
    word_count = 5
    workers = 4
    context = 5

    # 3. process the corpus
    print("Started lemmatization.")
    sentences = preprocessing.prepare_for_w2v(corpus, lemmatize,
                                              keep_stopwords)
    print("Finished lemmatization.")

    # 4. start learning
    info = "Started training with {} features, {} minimum word count, ".format(
        features, word_count)
    info2 = "{} threads, {} context window, default downsampling.".format(
        workers, context)
    print(info + info2)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    start_time = time.time()
    model = word2vec.Word2Vec(sentences, size=features, window=context,
                              min_count=word_count, workers=workers,
                              sg=1)

    end_time = time.time()
    print("Finished training in {:.3f} seconds.".format(end_time - start_time))

    # 5. save model
    if save:
        save_model(model, features, word_count, context,
                   lemmatize, book_count, keep_stopwords)

    print("Workflow completed.")
    return model


def save_model(model, num_features, min_word_count, context, lemmatized,
               book_count, keep_stopwords):
    """Save a word2vec model."""
    model_name = "saved_models/{}feat_{}minwords_{}context_{}books"\
                 .format(num_features, min_word_count, context, book_count)

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
