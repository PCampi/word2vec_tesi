"""Test for the postproc module."""
import unittest
import postproc


class PostprocTestCase(unittest.TestCase):
    """Test for `postproc.py`"""

    def test_filter_duplicates(self):
        """Test the `filter_duplicates` function."""
        test_data = [('xyz', 0.5),
                     ('yzx', 0.7),
                     ('ciao pippo', 0.3),
                     ('ciao bello', 0.5),
                     ('yzx', 0.4),
                     ('ciao bello', 0.1)]
        result = postproc.filter_duplicates(test_data)

        self.assertEqual(len(result), 4)

        self.assertEqual(result[0][0], 'xyz')
        self.assertEqual(result[1][0], 'yzx')
        self.assertEqual(result[2][0], 'ciao pippo')
        self.assertEqual(result[3][0], 'ciao bello')

        self.assertEqual(result[0][1], 0.5)
        self.assertEqual(result[1][1], 0.7)
        self.assertEqual(result[2][1], 0.3)
        self.assertEqual(result[3][1], 0.5)
