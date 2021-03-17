import unittest

from data.wikipedia.extract_pages_metadata import remove_within


class TestRemoveWithin(unittest.TestCase):

    def setUp(self):
        self.tag = 'tag'
        self.expected = 'Short test string'

    def test_same(self):
        actual = remove_within(self.expected, self.tag)
        self.assertEqual(actual, self.expected)

    def test_single(self):
        s = 'Short test<tag>hidden</tag> string'
        actual = remove_within(s, self.tag)
        self.assertEqual(actual, self.expected)

    def test_multiple(self):
        s = 'Short<tag>hidden1</tag> test<tag>hidden2</tag> string'
        actual = remove_within(s, self.tag)
        self.assertEqual(actual, self.expected)

    def test_multiple_with_other_tags(self):
        s = 'Short<tag>hidden1<othertag>something</othertag></tag> test<tag>hidden2</tag> string'
        actual = remove_within(s, self.tag)
        self.assertEqual(actual, self.expected)


if __name__ == '__main__':
    unittest.main()
