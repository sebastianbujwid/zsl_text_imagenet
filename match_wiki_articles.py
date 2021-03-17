import h5py
import yaml
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path

from definitions import ROOT_DIR
from utils import SimpleEnv
from data import imagenet
from data.wikipedia import WikiGraph
from matching.wiki_matching import match_titles, match_categories
from matching.wiki_matching import get_processes_wiki_titles, load_wiki_pages_structure, filter_titles
from matching.wiki_matching import get_all_discard_categories


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT_DIR / 'configs' / 'matching_config.yml')
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'configs' / 'env_config.yml')
    parser.add_argument('--imagenet_split', type=str, default='trainval_classes')
    parser.add_argument('--output_dir', type=Path, default=Path.cwd())
    parser.add_argument('--overwrite', nargs='+')
    return parser.parse_args()


def get_imagenet_structures(env_config, imagenet_split):
    with h5py.File(env_config.datasets.ImageNet.data_splits_file, 'r') as f:
        imagenet_ids = np.array(f[imagenet_split], dtype=np.int32)

    imagenet_id_details = imagenet.extract_imagenet_id_details(env_config.datasets.ImageNet.imagenet_ids_to_wordnet)

    imagenet_wordnet_ancestors = pickle.load(
        open(ROOT_DIR / 'data' / 'imagenet' / 'imagenet_wordnet_ancestor_categories.pkl', 'rb')
    )
    wordnet_to_wiki_categories = yaml.load(
        open(ROOT_DIR / 'matching' / 'data' / 'wordnet_to_wiki_categories.yml', 'r')
    )

    return imagenet_ids, imagenet_id_details, imagenet_wordnet_ancestors, wordnet_to_wiki_categories


def main(args):
    exp_env = SimpleEnv(args, output_dir=args.output_dir, configs=[args.config, args.env_config],
                        overwrite_config=args.overwrite)
    config = exp_env.config
    matching_config = config.matching
    env_config = config.env

    wiki_pages_structure = load_wiki_pages_structure(env_config.Wikipedia.wiki_pages_structure)
    wiki_graph = WikiGraph(page_to_categories=wiki_pages_structure['page_categories'],
                           category_parents=wiki_pages_structure['category_parents'])
    discard_categories = get_all_discard_categories(wiki_graph.category_parents.keys())

    imagenet_ids, imagenet_id_details, imagenet_wordnet_ancestors, wordnet_to_wiki_categories \
        = get_imagenet_structures(env_config, imagenet_split=args.imagenet_split)

    processed_wiki_titles = get_processes_wiki_titles(wiki_pages_structure)
    del wiki_pages_structure    # NOTE: save memory

    matched_articles = {}
    unmatched_articles = {}
    logging.info('\n\nMatching classes...')
    for imagenet_id in imagenet_ids:
        imagenet_class = imagenet_id_details[imagenet_id]
        wnid, wordnet_phrases = imagenet_class
        candidate_wiki_titles = match_titles(imagenet_class=imagenet_class,
                                             processed_titles=processed_wiki_titles)

        candidate_wiki_titles = filter_titles(titles=candidate_wiki_titles,
                                              discard_categories=discard_categories,
                                              wiki_graph=wiki_graph,
                                              max_depth=matching_config.discard_categories.max_depth)

        matched_wiki_titles = None

        titles_with_right_cats_matched = {}
        titles_with_wrong_cats_matched = {}
        for title in candidate_wiki_titles:
            matched_right_cats, matched_wrong_cats = match_categories(
                title, imagenet_class,
                wordnet_to_wiki_categories,
                wordnet_ancestors=imagenet_wordnet_ancestors[wnid],
                wiki_graph=wiki_graph,
                max_depth=matching_config.match_categories.max_depth)

            if len(matched_right_cats) > 0:
                titles_with_right_cats_matched[title] = (matched_right_cats, matched_wrong_cats)
            if len(matched_wrong_cats) > 0:
                titles_with_wrong_cats_matched[title] = (matched_right_cats, matched_wrong_cats)

        if len(titles_with_right_cats_matched) == 1:
            matched_title = list(titles_with_right_cats_matched.keys())[0]
            matched_wiki_titles = [matched_title]

        if matched_wiki_titles is None:
            unmatched_articles[imagenet_id] = {
                'wnid': wnid,
                'phrases': wordnet_phrases,
                'candidate_titles': candidate_wiki_titles,
            }
        else:
            matched_articles[imagenet_id] = {
                'wnid': wnid,
                'phrases': wordnet_phrases,
                'matched_titles': matched_wiki_titles,
            }

    logging.info(f'Matched: {len(matched_articles)}/{len(imagenet_ids)} classes!')

    logging.info(f'Matched: {len(matched_articles)}')
    logging.info(f'Unmatched: {len(unmatched_articles)}')

    results_file_suffix = f'imagenet-{args.imagenet_split}' \
                          f'_{Path(env_config.Wikipedia.wiki_pages_structure).stem.replace("pages_structure_", "")}' \
                          f'.pkl'
    matched_file = exp_env.run_dir / f'matches_{results_file_suffix}'
    logging.info(f'Saving matches to: {matched_file}')
    pickle.dump(matched_articles, open(matched_file, 'wb'))

    unmatched_file = exp_env.run_dir / f'unmatched_{results_file_suffix}'
    logging.info(f'Saving umatched to: {unmatched_file}')
    pickle.dump(unmatched_articles, open(unmatched_file, 'wb'))

    logging.info('Done! Existing')


if __name__ == '__main__':
    main(parse_args())
