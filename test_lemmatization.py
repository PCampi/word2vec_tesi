"""Lemmatization test."""
import unittest
import lemmatization


class LemmatizationTestCase(unittest.TestCase):
    """Test for `lemmatization.py`"""
    def setUp(self):
        self.lemmatizer = lemmatization.MyLemmatizer()

    def tearDown(self):
        self.lemmatizer = None

    def test_nouns(self):
        """Test that it lemmatizes correctly some easy words."""
        lemmas = ['cat', 'horse', 'house']
        for l in lemmas:
            computed = self.lemmatizer.smart_lemmatize(l)
            self.assert_lemma(computed, l)

    def test_verbs(self):
        """Test that it lemmatizes verbs correctly."""
        verbs = ['going', 'looked', 'went', 'are', 'did', 'bitten']
        lemmas = ['go', 'look', 'go', 'be', 'do', 'bite']

        for i in range(len(verbs)):
            self.assert_lemma(self.lemmatizer.smart_lemmatize(verbs[i]),
                              lemmas[i])

    def assert_lemma(self, computed, exact):
        """Assert that a computed word is correctly assigned lemma."""
        self.assertEqual(computed, exact,
                         "Wrong computed lemma {} for word {}".format(computed,
                                                                      exact))


if __name__ == "__main__":
    unittest.main()
