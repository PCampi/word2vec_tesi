from nltk.stem import WordNetLemmatizer


class MyLemmatizer(WordNetLemmatizer):
    """Class extending WordNetLemmatizer with smart lemmatization."""

    def __init__(self):
        super()

    def smart_lemmatize(self, word):
        """Lemmatize a word."""
        lemma = super.lemmatize(word)
        if lemma != word:
            return lemma

        # altrimenti prova con verbi, aggettivi
        else:
            # prova col verbo
            verb = super.lemmatize(word, pos='v')
            if verb != word:
                return verb

            adj = super.lemmatize(word, pos='a')
            if adj != word:
                return adj

            adv = super.lemmatize(word, pos='r')
            if adv != word:
                return adv

            return word
