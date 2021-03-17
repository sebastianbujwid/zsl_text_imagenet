import re
import tqdm
import yaml
import pickle
import logging
import collections
from typing import Mapping, Set, Tuple, List

from definitions import ROOT_DIR
from data import imagenet
from data.wikipedia import WikiGraph


def load_wiki_pages_structure(pkl_file):
    with open(pkl_file, 'rb') as f:
        d = pickle.load(f)
    # category_redirects = d['category_redirects']
    return {
        'redirects': d['redirects'],
        'page_categories': d['page_categories'],
        'category_parents': d['category_parents']
    }


def get_processes_wiki_titles(wiki_pages_structure):
    redirects = wiki_pages_structure['redirects']

    # wiki_titles = list(wiki_meta.keys())
    wiki_titles = set(wiki_pages_structure['page_categories'].keys())

    # Multiple titles can be the same after being processed,
    # need to be able to trace back the original unprocessed title
    logging.info(f'Processing titles')
    processed_wiki_titles = collections.defaultdict(set)
    for title in tqdm.tqdm(wiki_titles, unit='pages'):
        processed_wiki_titles[process_title(title)].add(title)

    logging.info('Processing redirect titles')
    for from_title in tqdm.tqdm(redirects.keys(), unit='redirects'):
        redirect_target = imagenet.find_redirect_target(from_title, wiki_titles, redirects)
        processed_wiki_titles[process_title(from_title)].add(redirect_target)
    return processed_wiki_titles


def process_title(title):
    title = title.lower()
    title = re.sub(r'\s*\(.+\)\s*$', '', title)  # NOTE: Can get less careful matching without this line
    return title


def match_titles(imagenet_class,  # tuple: (wnid, phrases)
                 processed_titles: Mapping[str, set]  # processed_title -> set of original (unprocessed titles)
                 ) -> Set:
    wnid, phrases = imagenet_class
    assert isinstance(phrases, str) or isinstance(phrases, list)
    single_phrase = isinstance(phrases, str)

    if single_phrase:
        p_phrase = process_title(title=phrases)
        if p_phrase in processed_titles:
            matched_original_titles = processed_titles[p_phrase]
            return matched_original_titles

        else:
            logging.warning(f'No match found for: {wnid}, {phrases}')
            return set()

    else:  # Multiple phrases
        p_phrases = [process_title(title=p) for p in phrases]
        matches = [p in processed_titles for p in p_phrases]
        matched_original_titles = set()
        for p_phrase, match in zip(p_phrases, matches):
            if not match:
                continue
            else:
                matched_original_titles.update(processed_titles[p_phrase])

        if len(matched_original_titles) > 0:
            return matched_original_titles
        else:
            logging.warning(f'No match found for: {wnid}, {phrases}')
            return set()


def match_categories(wiki_title, imagenet_class,
                     wordnet_to_wiki_categories: Mapping,
                     wordnet_ancestors: Set,
                     wiki_graph: WikiGraph,
                     max_depth: int,
                     ):
    wiki_categories_should_be_matched = set()
    for wnid in wordnet_ancestors:
        if wnid not in wordnet_to_wiki_categories:
            continue
        c = wordnet_to_wiki_categories[wnid]
        if isinstance(c, list):
            wiki_categories_should_be_matched.update(c)
        else:
            wiki_categories_should_be_matched.add(c)

    if len(wiki_categories_should_be_matched) == 0:
        logging.error(f'No wiki categories for {imagenet_class}! WordNet ancestors: {wordnet_ancestors}')

    all_target_categories = set()
    for c in wordnet_to_wiki_categories.values():
        if isinstance(c, str):
            all_target_categories.add(c)
        elif isinstance(c, list):
            all_target_categories.update(c)
        else:
            raise NotImplementedError()

    wiki_categories_should_not_be_matched = all_target_categories.difference(wiki_categories_should_be_matched)

    all_reachable_wiki_cats = wiki_graph.reachable_ancestor_categories(wiki_title, max_depth)

    matched_right = all_reachable_wiki_cats.intersection(wiki_categories_should_be_matched)
    matched_wrong = all_reachable_wiki_cats.intersection(wiki_categories_should_not_be_matched)

    return matched_right, matched_wrong


def evaluate_matches(matches: Mapping[int, List[str]],
                     true_matches: Tuple[Mapping, Set]  # (matches, without_matches)
                     ):
    """
    A class is considered true positive if it doesn't contain any false positive titles
    """
    true_matched, true_no_matches = true_matches

    false_positive_classes_titles = {}

    # Calculate precision
    true_positive_classes = 0
    num_positive_classes = 0
    for imagenet_id, true_titles in true_matched.items():
        assert isinstance(true_titles, list)
        if imagenet_id not in matches:
            continue

        matched_titles = {m['title'] for m in matches[imagenet_id]['matches']}
        false_positive_titles = matched_titles.difference(set(true_titles))
        contains_false_positive_titles = len(false_positive_titles)
        if contains_false_positive_titles:
            false_positive_classes_titles[imagenet_id] = false_positive_titles
        else:
            true_positive_classes += 1
        num_positive_classes += 1

    for imagenet_id in true_no_matches:
        if imagenet_id not in matches:
            continue
        else:
            false_positive_classes_titles[imagenet_id] = matches[imagenet_id]
            num_positive_classes += 1

    logging.info(f'Precision: {true_positive_classes}/{num_positive_classes}'
                 f' = {true_positive_classes / num_positive_classes}')
    return false_positive_classes_titles


def filter_titles(titles, discard_categories: Set, max_depth: int, wiki_graph: WikiGraph) -> Set:
    filtered_titles = []
    for t in titles:
        reachable_cats = wiki_graph.reachable_ancestor_categories(t, max_depth)
        if len(reachable_cats.intersection(discard_categories)) == 0:
            filtered_titles.append(t)
    return set(filtered_titles)


def is_any_in(vals, x):
    for v in vals:
        if v in x:
            return True

    return False


def filter_discard_categories(c):
    ignore = {
        'film', 'album', 'games', 'comics', 'songs', 'music genres',
        'magazines', 'book series', 'books', 'novels', 'science fiction', 'literature',
        'fiction', 'monsters', 'creatures',
        'works',
        'people', 'births', 'deaths',
        'by manufacturer', 'companies',
        'software', 'viruses',
        'national parks',
    }
    keep = {
        'breeds', 'animal', 'plants'
    }

    x = c.lower()

    if is_any_in(keep, x):
        return False

    if is_any_in(ignore, x):
        return True

    return False


def load_discard_categories():
    discard_categories = yaml.load(
        open(ROOT_DIR / 'matching' / 'data' / 'discard_categories_filtered.yml', 'r')
    )
    return set(discard_categories['DISCARD'])


def get_all_discard_categories(all_categories):
    discard_categories = load_discard_categories()
    filtered_discard_cats = list(filter(filter_discard_categories, all_categories))
    discard_categories.update(filtered_discard_cats)
    discard_categories.update([
        'Category:Given names',
    ])
    return discard_categories
