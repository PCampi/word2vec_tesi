"""Test for the workflow module."""
import unittest
import pickle

# import persistence
# import workflow
import emolex_persistence


class WorkflowTestCase(unittest.TestCase):
    """Test for `workflow.py`"""

    def test_not_in_emolex(self):
        """Test for the function `filter_not_in_emolex`."""
        emolex = emolex_persistence.load_emolex("data/EmoLex_en.pickle")

        with open("results/not_in_emolex.pickle", 'rb') as f:
            not_in_emolex = pickle.load(f)

        words_to_test = [word for key in not_in_emolex
                         for word, _ in not_in_emolex[key]]

        for word in words_to_test:
            self.assertFalse(word in emolex)

    # def test_in_emolex(self):
    #     """Test that the filtered lemmas are in emolex."""
    #     with open("results/lemmas_filtered.pickle", "rb") as f:
    #         lemmas_filtered = pickle.load(f)

    #     emolex = emolex_persistence.load_emolex("data/EmoLex_en.pickle")

    #     words_to_test =
