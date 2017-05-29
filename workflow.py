from gensim.models import KeyedVectors
import pudb

import emolex_persistence
import lemmatization
import persistence
import postproc
import similarity


def compute_most_similar(words, model, number=50):
    """Compute the N most similar words."""
    model = KeyedVectors.load("model_stripped")
    it_limit = 20

    result = {
        w: similarity.get_most_similar(w, model, how_many=number,
                                       max_iter=it_limit)
        for w in words}

    return result


def compute_lemmas(tpl_arr, lemmatizer):
    """Compute the lemmas of the words using the provided lemmatizer."""
    return [(lemmatizer.smart_lemmatize(word), score)
            for word, score in tpl_arr]


def filter_not_in_emolex(lem_dict, emolex_dict):
    """For every array of tuples, leave only the missing EmoLex entries.

    Parameters
    ----------
    lem_dict: dictionary
        contains {emotion: [(word, score), (word, score)], ...}

    Returns
    -------
    not_in_emolex: dictionary
        contains {emotion: [word_not_in_emolex, ...], ...}
    """
    # pudb.set_trace()

    not_in_emolex = {emotion: [tpl for tpl in
                               filter(
                                   (lambda tpl:
                                    tpl[0].lower() not in emolex_dict),
                                   lem_dict[emotion])]
                     for emotion in lem_dict}

    return not_in_emolex


def annotate(lem_dict, emolex_dict):
    annotated = {e: [(w, s, w in emolex_dict) for w, s in lem_dict[e]]
                 for e in lem_dict}
    return annotated


def _setup():
    """Convenience initialization function."""
    model_file = "model_stripped"
    model = KeyedVectors.load(model_file)

    emotions = ["anger", "anticipation", "disgust", "fear", "joy",
                "sadness", "surprise", "trust"]
    top = 50
    coll = compute_most_similar(emotions, model, number=top)
    return coll


if __name__ == "__main__":
    coll = _setup()

    lemmatizer = lemmatization.MyLemmatizer()
    lemmas = {e: compute_lemmas(coll[e], lemmatizer) for e in coll}

    lemmas_no_duplicates = {e: postproc.filter_duplicates(lemmas[e])
                            for e in lemmas}

    emolex = emolex_persistence.load_emolex("data/EmoLex_en.pickle")
    not_in_emolex = filter_not_in_emolex(lemmas_no_duplicates, emolex)

    persistence.save_results(not_in_emolex, "not_in_emolex.pickle")
    persistence.save_results(not_in_emolex, "not_in_emolex.csv")
