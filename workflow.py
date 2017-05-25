from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer

import emolex_persistence
import postproc
import prova


def filter_not_in_emolex(lem_dict):
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
    not_in_emolex = {emotion: [tpl for tpl in
                               filter(
                                   (lambda tpl: tpl[0].lower() not in emolex),
                                   emolex[emotion])]
                     for emotion in lem_dict}

    return not_in_emolex


if __name__ == "__main__":
    model_file = "model_stripped"
    model = KeyedVectors.load(model_file)

    emotions = ["anger", "anticipation", "disgust", "fear", "joy",
                "sadness", "surprise", "trust"]
    top = 50
    it_limit = 20
    coll = {
        e: prova.get_most_similar(e, model, how_many=top, max_iter=it_limit)
        for e in emotions}

    lemmatizer = WordNetLemmatizer()
    lemmas = {e: [(postproc.smart_lemmatize(word, lemmatizer), score)
                  for word, score in coll[e]] for e in coll}

    lemmas_filtered = {e: postproc.filter_duplicates(lemmas[e])
                       for e in lemmas}

    emolex = emolex_persistence.load_emolex("data/EmoLex_en.pickle")
    not_in_emolex = filter_not_in_emolex(lemmas_filtered)

    postproc.save_results("not_in_emolex.pickle")
