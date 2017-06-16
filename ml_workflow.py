"""Workflow for the machine learning part with Word2Vec."""
import time
from typing import List

from gensim.models import word2vec
import preprocessing

import logging


def preprocess_file(path, keep_stopwords=False, lemmatize=True):
    """Read the contents of a file and prepare it for w2v."""
    with open(path, 'r') as f:
        text = f.read()

    cleaned_text = preprocessing.prepare_for_w2v(text, keep_stopwords,
                                                 lemmatize)
    return cleaned_text


def train_model(sentences: List[List[str]], num_features=300,
                min_word_count=40, num_workers=4, context=10,
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
    print("Started training.")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    start_time = time.time()
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)

    end_time = time.time()
    print("Finished training in {:.3f} seconds.".format(end_time - start_time))
    return model


def complete_workflow(save=True, keep_stopwords=True, lemmatize=False):
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
        model_name = "{}feat_{}minwords_{}context".\
                     format(features, word_count, context)
        if lemmatize:
            model_name = model_name + "_lemmatized"
        if keep_stopwords:
            model_name = model_name + "_with_stopwords"
        else:
            model_name = model_name + "_no_stopwords"

        model.save(model_name)
        print("Model saved! Workflow completed.")
    else:
        print("Workflow completed.")

    return model


if __name__ == "__main__":
    complete_workflow(save=True)
