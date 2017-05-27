from nltk.stem import WordNetLemmatizer


class MyLemmatizer(WordNetLemmatizer):
    """Class extending WordNetLemmatizer with smart lemmatization."""

    def __init__(self):
        super()

    def smart_lemmatize(self, word):
        """Lemmatize a word."""
        lemma = self.lemmatize(word)
        if lemma != word:
            return lemma

        # altrimenti prova con verbi, aggettivi
        else:
            # prova col verbo
            verb = self.lemmatize(word, pos='v')
            if verb != word:
                return verb

            adj = self.lemmatize(word, pos='a')
            if adj != word:
                return adj

            adv = self.lemmatize(word, pos='r')
            if adv != word:
                return adv

            return word
