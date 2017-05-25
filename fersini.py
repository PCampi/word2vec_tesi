from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
import pickle

import prova
import postproc

model_file = "model_stripped"
model = KeyedVectors.load(model_file)

emotions = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

top = 50
it_limit = 20

coll = {e: prova.get_most_similar(e,
                                  model,
                                  how_many=top,
                                  max_iter=it_limit)
        for e in emotions}

# Now load emolex and lemmatizer
lemmatizer = WordNetLemmatizer()
lemmas = {e: [(postproc.lemmatize(word, lemmatizer), score)
              for word, score in coll[e]]
          for e in coll}

lemmas_filtered = {e: postproc.filter_duplicates(lemmas[e])
                   for e in lemmas}

with open("data/EmoLex_en.pickle", 'rb') as f:
    emolex = pickle.load(f)


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
