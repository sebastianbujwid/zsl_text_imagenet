import pickle
import os
import logging
import itertools
import collections
from pathlib import Path

from data.graph_algorithms import compute_ancestors

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.INFO)


DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def parse_wordnet_hierarchy(file):
    with open(file, 'r') as f:
        parent_child_lines = f.read().splitlines()
    parent_child_pairs = [x.split(' ') for x in parent_child_lines]
    return parent_child_pairs


def main():
    wornet_hierarchy_file = DIR / 'wordnet.is_a.txt'
    parent_child_pairs = parse_wordnet_hierarchy(wornet_hierarchy_file)
    imagenet_wordnet_ancestor_categories_file = DIR / 'imagenet_wordnet_ancestor_categories.pkl'
    if imagenet_wordnet_ancestor_categories_file.exists():
        raise ValueError(f'File {imagenet_wordnet_ancestor_categories_file} already exists!')

    category_parents = collections.defaultdict(set)
    for parent, cat in parent_child_pairs:
        category_parents[cat].add(parent)

    all_categories = set(itertools.chain(*parent_child_pairs))

    category_ancestors = compute_ancestors(all_categories, category_parents)

    pickle.dump(category_ancestors, open(imagenet_wordnet_ancestor_categories_file, 'wb'),
                protocol=4)


if __name__ == '__main__':
    main()
