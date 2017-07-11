# coding: utf-8

"""Enhance EmoLex using results from the ANN."""
import numpy as np
from functools import reduce

mapping = {'rabbia': 0, 'anticipazione': 1, 'disgusto': 2, 'paura': 3,
           'gioia': 4, 'tristezza': 5, 'sorpresa': 6, 'fiducia': 7}


def add_to_emolex(word, emotion, emolex):
    """Add an emotion to a word in emolex."""
    modified_words = set()
    try:
        e_lst = emolex[word]
        if len(e_lst) == 1:
            emotions = e_lst[0]
            index = mapping[emotion]
            if emotions[index] == 0:
                emotions[index] = 1
                emolex[word] = [emotions]  # add to emolex the new emotion
                modified_words.add(word)
                print("Aggiungo {} alla parola {}".format(emotion, word))
            else:
                print("{} possiede già {} in emolex".format(word, emotion))
        elif len(e_lst) > 1:
            emotions = reduce(np.logical_or, e_lst).astype(np.int16)
            index = mapping[emotion]
            if emotions[index] == 0:
                e_lst[0][index] = 1
                emolex[word] = e_lst  # add to emolex the new emotion
                modified_words.add(word)
                print("Aggiungo {} alla parola {}".format(emotion, word))
            else:
                print("{} possiede già {} in emolex".format(word, emotion))
    except KeyError:
        new_word_array = np.zeros(8, dtype=np.int16)
        new_word_array[mapping[emotion]] = 1
        emolex[word] = [new_word_array]
        modified_words.add(word)
        print("Aggiungo nuova parola {} a emolex e le assegno {}"
              .format(word, emotion))

    return modified_words
