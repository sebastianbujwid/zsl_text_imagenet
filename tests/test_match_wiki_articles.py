import unittest

from match_wiki_articles import match_titles
from matching.wiki_matching import process_title


class TestProcessTitle(unittest.TestCase):

    def test_no_parentheses(self):
        t = "Tree"
        expected = 'tree'
        actual = process_title(t)
        self.assertEqual(actual, expected)

    def test_ends_with_parentheses(self):
        expected = "tree"
        for t in ["Tree (data structure)", "Tree (graph theory)", "Tree (surname)"]:
            actual = process_title(t)
            self.assertEqual(actual, expected)

    def test_middle_parentheses(self):
        t = "Greenville (CDP), New York"
        expected = 'greenville (cdp), new york'
        actual = process_title(t)
        self.assertEqual(actual, expected)


class TestMatchTitle(unittest.TestCase):

    def setUp(self):
        self.processed_titles = {
            'small house': {'Small house', 'Small House', 'House', 'house'},
            'robot': {'Robot'},
            'android': {'Robot'},
        }
        self.wnid = 'n1'

    def test_single_match(self):
        for query in ['ROBOT', 'robot', 'Robot']:
            matched = match_titles(
                imagenet_class=(self.wnid, query),
                processed_titles=self.processed_titles
            )

            expected = {'Robot'}
            self.assertEqual(matched, expected)

    def test_multiple_phrases_single_match(self):
        queries = [
            ['robot', 'android'],
            ['Robot', 'Android'],
        ]
        for query in queries:
            matched = match_titles(
                imagenet_class=(self.wnid, query),
                processed_titles=self.processed_titles
            )
            expected = {'Robot'}
            self.assertEqual(matched, expected)

    def test_multiple_phrases_same_original_title(self):
        queries = [
            ['robot', 'small robot'],
            ['Robot', 'Small robot'],
        ]
        for query in queries:
            matched = match_titles(
                imagenet_class=(self.wnid, query),
                processed_titles=self.processed_titles
            )
            expected = {'Robot'}
            self.assertEqual(matched, expected)

    def test_fail_no_match(self):
        for query in ['', 'title', 'rrobot']:
            matched = match_titles(
                imagenet_class=(self.wnid, query),
                processed_titles=self.processed_titles
            )
            expected = set()
            self.assertEqual(matched, expected)

    def test_multiple_original(self):
        for query in ['Small House', 'small house', 'Small house', 'SMALL HOUSE']:
            matched = match_titles(
                imagenet_class=(self.wnid, query),
                processed_titles=self.processed_titles
            )
            expected = {'Small house', 'Small House', 'House', 'house'}
            self.assertEqual(matched, expected)

    def test_multiple_matches(self):
        queries = [
            ['Small House', 'Robot'],
            ['small house', 'robot'],
        ]
        for query in queries:
            matched = match_titles(
                imagenet_class=(self.wnid, query),
                processed_titles=self.processed_titles
            )
            expected = {'Robot', 'Small house', 'Small House', 'House', 'house'}
            self.assertEqual(matched, expected)

    def test_multiple_phrases_no_match(self):
        queries = [
            ['Tree', 'Small tree'],
            ['Stone', 'Small stone']
        ]
        for query in queries:
            matched = match_titles(
                imagenet_class=(self.wnid, query),
                processed_titles=self.processed_titles
            )
            expected = set()
            self.assertEqual(matched, expected)


if __name__ == '__main__':
    unittest.main()
