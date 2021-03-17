import re
import csv
import pickle
import logging

from data import imagenet


def remove_comments(text):
    comment_pttrn = r'#.*'
    text = re.sub(comment_pttrn, '', text)
    return text


def load_matches(dataset_config, expected_imagenet_ids=None):
    logging.info('Loading matches...')
    matches = load_automatic_matches(dataset_config)
    manual_matches, manual_without_matches = load_manual_matches(dataset_config)

    imagenet_id_details = imagenet.extract_imagenet_id_details(dataset_config['imagenet_ids_to_wordnet'])

    for imagenet_id, titles in manual_matches.items():
        if imagenet_id in matches:
            if matches[imagenet_id]['matched_titles'] != manual_matches[imagenet_id]:
                logging.warning(f'Overwriting ImageNet ID: {imagenet_id},'
                                f' {matches[imagenet_id]} -> {manual_matches[imagenet_id]}')
            matches[imagenet_id]['matched_titles'] = manual_matches[imagenet_id]
        else:
            wnid, phrases = imagenet_id_details[imagenet_id]
            matches[imagenet_id] = {
                'wnid': wnid,
                'phrases': phrases,
                'matched_titles': titles
            }

    for imagenet_id in manual_without_matches:
        if imagenet_id in matches:
            logging.warning(f'Overwriting ImageNet ID: {imagenet_id},'
                            f' {matches[imagenet_id]} -> None')
            del matches[imagenet_id]

    logging.info(f'Loaded {len(matches)} matches\n')

    missing_classes = 0
    if expected_imagenet_ids:
        for imagenet_id in expected_imagenet_ids:

            if imagenet_id not in matches and imagenet_id not in manual_without_matches:
                logging.warning(f'ImageNet ID: {imagenet_id}, {imagenet_id_details[imagenet_id]}'
                                f' has no match and has not been set as an empty match')
                missing_classes += 1

    assert len(expected_imagenet_ids) == (len(matches) + missing_classes + len(manual_without_matches))

    return matches


def load_automatic_matches(dataset_config):
    if 'matched_pages_pkl' in dataset_config and dataset_config['matched_pages_pkl']:
        matched_pages_pkl = dataset_config['matched_pages_pkl']
        with open(matched_pages_pkl, 'rb') as f:
            matched_pages = pickle.load(f)
    else:
        logging.warning(f'No "matched_pages_pkl" defined, using only csv matches')
        matched_pages = {}

    return matched_pages


def load_manual_matches(dataset_config):
    wordnet_to_imagenet_id = imagenet.extract_wordnet_to_imagenet_id(dataset_config['imagenet_ids_to_wordnet'])

    manual_matches_csv = dataset_config['manual_matches_csv']
    matches = {}
    without_matches = set()
    logging.info(f'Extracting articles for manually matched articles')
    with open(manual_matches_csv, 'r') as f:
        for line in f:
            line = remove_comments(line)
            line = line.strip()
            if len(line) == 0:
                continue

            split = re.split(r'\s*,\s*', line)
            wnid, wiki_titles = split[0], split[1:]

            imagenet_id = wordnet_to_imagenet_id[wnid]
            if imagenet_id in matches or imagenet_id in without_matches:
                raise ValueError(f'"{wnid}" present multiple times in manual matches!')

            has_no_article = len(wiki_titles) and (wiki_titles[0] == '_' or wiki_titles[0] == '-')
            if has_no_article:
                without_matches.add(imagenet_id)
            else:
                assert isinstance(wiki_titles, list)
                matches[imagenet_id] = wiki_titles

    return matches, without_matches


def get_wnid(noun_id):
    return f'n{int(noun_id):08d}'


def load_correspondences_the_peoples_web(env_config, select_imagenet_ids):
    wordnet_to_imagenet_id = imagenet.extract_wordnet_to_imagenet_id(
        env_config.datasets.ImageNet['imagenet_ids_to_wordnet'])
    imagenet_id_details = imagenet.extract_imagenet_id_details(env_config.datasets.ImageNet.imagenet_ids_to_wordnet)

    select_wnids = set([imagenet_id_details[i][0] for i in select_imagenet_ids])

    brief_tsv = env_config.WordNet_Wikipedia_alignment.the_peoples_web_meets_linguistic_knowledge_brief_file

    matches = {}
    with open(brief_tsv, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            noun_id, title = row
            wnid = get_wnid(noun_id)
            if not select_wnids or wnid in select_wnids:
                imagenet_id = wordnet_to_imagenet_id[wnid]
                matches[imagenet_id] = {
                    'wnid': wnid,
                    'phrases': imagenet_id_details[imagenet_id][1],
                    'matched_titles': [title]
                }

    return matches


def load_correspondences_dijkstra_wsa(env_config, select_imagenet_ids):
    wordnet_to_imagenet_id = imagenet.extract_wordnet_to_imagenet_id(
        env_config.datasets.ImageNet['imagenet_ids_to_wordnet'])
    imagenet_id_details = imagenet.extract_imagenet_id_details(env_config.datasets.ImageNet.imagenet_ids_to_wordnet)

    select_wnids = set([imagenet_id_details[i][0] for i in select_imagenet_ids])

    file = env_config.WordNet_Wikipedia_alignment.dijkstra_wsa_file

    matches = {}
    with open(file, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            prefix = '[POS: noun] '
            x, title = row
            assert x.startswith(prefix)
            noun_id = x.strip(prefix)

            wnid = get_wnid(noun_id)
            if not select_wnids or wnid in select_wnids:
                imagenet_id = wordnet_to_imagenet_id[wnid]
                matches[imagenet_id] = {
                    'wnid': wnid,
                    'phrases': imagenet_id_details[imagenet_id][1],
                    'matched_titles': [title]
                }

    return matches

