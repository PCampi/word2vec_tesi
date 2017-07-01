"""Preprocessing test."""
import unittest
import preprocessing as pp


class PreprocessingTestCase(unittest.TestCase):
    """Test for `preprocessing.py`"""

    def test_guillements_in_sentence(self):
        """Test that it correctly preprocess guillements."""
        with open("./test_data/test_guillements.txt", "r") as f:
            text = f.read()

        result = pp.guillements_in_sentence(text)
        expected = " Per molto tempo mi sono coricato presto la sera ." +\
                   "\nEcco l’inizio semplice e misterioso di questo primo" +\
                   " volume che apre la Recherche."

        self.assertEqual(result, expected, "Different results.")

    def test_rebalance_full_stops(self):
        text = "Nel mezzo. . . . . . . . . ."
        result = pp.rebalance_full_stops(text)
        expected = "Nel mezzo. "

        self.assertEqual(result, expected, "Different results")
