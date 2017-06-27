"""Workflow for the machine learning part with Word2Vec."""
from typing import List
import pickle
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


def read_all_corpus(train_dir="./train_data"):
    """Get all the text as a single string."""
    training_files = glob.glob(train_dir + '/*.txt')
    book_count = len(training_files)

    corpus = ""
    for f in training_files:
        with open(f, 'r') as new_file:
            corpus = corpus + new_file.read()

    return corpus, book_count


def create_cached_corpus(directory="./train_data"):
    """Pre-lemmatize all the corpus."""
    corpus, _ = read_all_corpus(directory)

    # lemmatize all the text
    start_time = time.time()
    print("Start lemmatization")

    sentences = preprocessing.prepare_for_w2v(corpus, lemmatize=True,
                                              keep_stopwords=False)
    end_time = time.time()
    elapsed = end_time - start_time
    print("Finished lemmatization in {}".format(elapsed))

    return sentences


def save_preprocessed_corpus(corpus: List[List[str]]):
    """Save the lemmatized and preprocessed corpus in a pickle file."""
    save_dir = "./train_data/corpus/"
    corpus_name = "corpus_{}.pickle".format(len(corpus))

    with open(save_dir + corpus_name, 'wb') as f:
        print("Saving corpus {}".format(corpus_name))
        pickle.dump(corpus, f)
        print("Save completed")


def complete_workflow(save=True, keep_stopwords=False, lemmatize=True,
                      train_dir="./train_data", use_cache=False,
                      cached_corpus_name=None):
    """A convenience function to run all the workflow.

    Parameters
    ----------
    save: boolean
        if true, save the model at the end of training

    keep_stopwords: boolean
        if true, keeps stopwords in the preprocessed text

    lemmatize: boolean
        if true, lemmatize the corpus

    train_dir: str
        relative path to the corpus collection directory

    use_cache: boolean
        if true, use the latest version of the preprocessed corpus

    cached_corpus_name: str
        name of the corpus file to use

    Returns
    -------
    model: gensim.model.word2vec
        the trained model
    """
    # 0. set the model parameters
    features = 200
    word_count = 5
    workers = 4
    context = 5
    # if not using cached corpus
    # 1. get the text to analyze
    book_count = None

    if not use_cache:
        corpus, book_count = read_all_corpus(train_dir)
        # 2. process the corpus
        print("Started lemmatization.")
        sentences = preprocessing.prepare_for_w2v(corpus, lemmatize,
                                                  keep_stopwords)
        print("Finished lemmatization.")
    else:
        corpus_path = train_dir + '/corpus/' + cached_corpus_name
        with open(corpus_path, 'rb') as c:
            sentences = pickle.load(c)

    # 3. start learning
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

    # 4. save model
    if save:
        if book_count is None:
            book_count = len(glob.glob(train_dir + '/*.txt'))

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


if __name__ == "__main__":
    complete_workflow(save=True)
