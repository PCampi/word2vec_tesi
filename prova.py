"""Prova di word2vec con Gensim."""

import pudb


def get_most_similar(word, model, how_many=50, negative=None, max_iter=5):
    """Get the how_many most similar words from the model."""
    n_ok = 0
    n = how_many + how_many // 4
    iter = 0

    while n_ok < how_many and iter < max_iter:
        if negative is not None:
            similar = model.most_similar(positive=[word],
                                         negative=negative,
                                         topn=n)
        else:
            similar = model.most_similar(positive=[word], topn=n)

        iter = iter + 1
        n = n + how_many // 4

        filtered_result = get_filtered(word, similar)
        n_ok = len(filtered_result)

    if n_ok < how_many:
        pudb.set_trace()
        raise Exception("Not enough words found in {} iterations"
                        .format(max_iter))

    return filtered_result[:how_many]


def is_unigram(tpl):
    """If is unigram return True."""
    word, _ = tpl
    return "_" not in word


def is_capitalized(w1, w2):
    return w2 == w1.capitalize() \
        or w1 == w2.capitalize()


def get_filtered(key, arr):
    """Filter leaving only unigrams and not capitalized version of key."""
    def is_not_capitalized(tpl):
        word, _ = tpl
        return not is_capitalized(key, word)

    unigrams = filter(is_unigram, arr)
    return list(filter(is_not_capitalized, (tpl for tpl in unigrams)))
