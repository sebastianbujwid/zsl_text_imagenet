import unittest

from data.graph_algorithms import reachable_category_ancestors_all


class TestReachableAncestors(unittest.TestCase):

    def setUp(self):
        self.loop = {
            'A': {'categories': ['B']},
            'B': {'categories': ['C']},
            'C': {'categories': ['A']},
        }

        self.loop_difficult = {
            '0': {'categories': ['A']},
            'A': {'categories': ['B', 'D']},
            'B': {'categories': ['C', 'E']},
            'C': {'categories': ['A']},
        }

        self.tree = {
            'C1': {'categories': ['B1']},
            'D1': {'categories': ['B1', 'B2']},
            'D2': {'categories': ['B2']},
            'B1': {'categories': ['A1']},
            'B2': {'categories': ['A2']},
        }

    def test_all_categories_are_returned(self):
        r, _ = reachable_category_ancestors_all({
            'A': {'categories': ['B']}
        })

        self.assertEqual(
            {
                'A': {'A', 'B'},
                'B': {'B'},
            },
            r
        )

    def test_loop(self):
        r, _ = reachable_category_ancestors_all(self.loop)

        self.assertEqual(
            {
                'A': {'A', 'B', 'C'},
                'B': {'A', 'B', 'C'},
                'C': {'A', 'B', 'C'},
            },
            r
        )

    def test_loop_difficult(self):
        r, _ = reachable_category_ancestors_all(self.loop_difficult)

        self.assertEqual(
            {
                '0': {'A', 'B', 'C', 'D', 'E', '0'},
                'A': {'A', 'B', 'C', 'D', 'E'},
                'B': {'A', 'B', 'C', 'D', 'E'},
                'C': {'A', 'B', 'C', 'D', 'E'},
                'D': {'D'},
                'E': {'E'},
            },
            r
        )

    def test_tree(self):
        r, _ = reachable_category_ancestors_all(self.tree)

        self.assertEqual(
            {
                'C1': {'C1', 'B1', 'A1'},
                'D2': {'D2', 'B2', 'A2'},
                'D1': {'D1', 'B1', 'A1', 'B2', 'A2'},
                'B1': {'B1', 'A1'},
                'B2': {'B2', 'A2'},
                'A1': {'A1'},
                'A2': {'A2'}
            },
            r
        )


if __name__ == '__main__':
    unittest.main()
