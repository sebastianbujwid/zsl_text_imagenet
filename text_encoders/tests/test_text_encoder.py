import unittest

from text_encoders.text_encoders_utils import split_with_overlap


class TestSplitWithOverlap(unittest.TestCase):

    def setUp(self):
        self.text = 'One two three four five. Six seven eight nine. Ten.'

    def test_output_type(self):
        output = split_with_overlap(self.text, 2, 1)
        self.assertEqual(type(output), list)

    def test_short(self):
        output = split_with_overlap(self.text, 100, 1)
        self.assertEqual(output, [self.text])

    def test_max_length(self):
        max_length = 2
        output = split_with_overlap(self.text, max_length=max_length, overlap_window_length=1)

        for x in output:
            actual = len(x.split())
            self.assertLessEqual(actual, max_length)

    def test_split(self):
        expected = [
            'One two three',
            'three four five.',
            'five. Six seven',
            'seven eight nine.',
            'nine. Ten.'
        ]
        output = split_with_overlap(self.text, max_length=3, overlap_window_length=1)
        self.assertEqual(output, expected)

    def test_split_2(self):
        expected = [
            'One two three four five.',
            'four five. Six seven eight',
            'seven eight nine. Ten.'
        ]
        output = split_with_overlap(self.text, max_length=5, overlap_window_length=2)
        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main()
