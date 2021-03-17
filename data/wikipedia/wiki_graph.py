import logging
from typing import Mapping, Any, Set

from data import graph_algorithms


class WikiGraph:

    def __init__(self,
                 page_to_categories,
                 category_parents: Mapping[Any, Mapping]
                 ):
        self.page_to_categories = page_to_categories
        self.category_parents = category_parents

    def _page_categories(self, title):
        if title not in self.page_to_categories:
            logging.error(f'"{title}" not in page_categories! Returning empty... ERROR!')
            return []

        return self.page_to_categories[title]['categories']

    def reachable_ancestor_categories(self, title, max_depth: int) -> Set:
        assert max_depth > 0

        page_categories = self._page_categories(title)
        reachable_categories = set(page_categories)

        if max_depth > 1:
            for category in page_categories:
                reachable_categories.update(
                    graph_algorithms.reachable_category_ancestors_single_category(category,
                                                                                  self.category_parents,
                                                                                  max_depth - 1)[0]
                )
        return reachable_categories

    def get_page_id(self, title: str):
        return self.page_to_categories[title]['id']
