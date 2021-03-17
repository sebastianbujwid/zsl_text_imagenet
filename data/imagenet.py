import re
import h5py
import json
import tqdm
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Mapping, Set
import xml.etree.ElementTree as ET

import matching
from data.wikipedia import readable_title
from definitions import ROOT_DIR


def get_imagenet_ids(split, dataset_config):
    with h5py.File(dataset_config['data_splits_file'], 'r') as f:
        split_imagenet_ids = set(np.array(f.get(split), dtype=np.int32))
    return split_imagenet_ids


def extract_wiki_article(wiki_id, wiki_title, meta_wiki, articles_path=None):
    article_meta = meta_wiki[wiki_title]
    assert article_meta['id'] == wiki_id

    article_file = Path(article_meta['file'])
    if articles_path is not None:
        assert article_file.parts[-1].startswith('wiki_')
        article_file = articles_path / article_file.relative_to(article_file.parent.parent)

    if not article_file.exists():
        raise RuntimeError(f'Could not find file: {article_file}')

    article_text = None
    with open(article_file, 'r', encoding='utf-8') as f:
        for i, l in enumerate(f):
            if len(l) < 1:
                continue
            try:
                article_json = json.loads(l)
            except:
                logging.error(f'Problem with reading: {article_file}, line: {i}')
                raise RuntimeError(f'Problem with reading: {article_file}, line: {i}')

            if int(article_json['id']) == wiki_id:
                assert readable_title(article_json['title']) == wiki_title
                article_text = article_json['text']
                break

    if article_text is None:
        raise ValueError(f'Could not find article: ({wiki_id}, {wiki_title}) in {article_file}')
    return article_text


def select_article_sections(article, filter_sections):
    if filter_sections == 'ALL' or filter_sections == ['ALL']:
        return article

    section_pttrn = r'(Section::::.*\n)'
    section_splits = re.split(section_pttrn, article)

    sections = []
    section = 'Section::::ABSTRACT.\n'
    section_text = None
    for x in section_splits:
        if re.match(section_pttrn, x) is not None:
            section = x
            section_text = None
        else:
            assert section_text is None
            section_text = x

            sections.append((section, section_text))

    filter_sections_strings = set([f'Section::::{x}.\n' for x in filter_sections])
    selected_sections = [(name, text) for name, text in sections if name in filter_sections_strings]

    joined_sections = ''.join(
        [''.join((name, text)) if name != 'Section::::ABSTRACT.\n' else text
         for name, text in selected_sections]
    )
    return joined_sections


def extract_imagenet_id_details(imagenet_w2v_extra_pkl):
    with open(imagenet_w2v_extra_pkl, 'rb') as f:
        dict_data = pickle.load(f)
    imagenet_details = {id + 1: (wnid, words) for id, (wnid, words) in enumerate(
        zip(dict_data['wnids'], dict_data['words'])
    )}
    return imagenet_details


def extract_wordnet_to_imagenet_id(imagenet_w2v_extra_pkl):
    with open(imagenet_w2v_extra_pkl, 'rb') as f:
        dict_data = pickle.load(f)
    wordnet_to_imagenet_id = {wnid: id+1 for id, wnid in enumerate(dict_data['wnids'])}
    return wordnet_to_imagenet_id


def find_redirect_target(from_title: str,
                         wiki_titles: Set,
                         redirects: Mapping[str, Mapping]
                         ) -> str:
    title = redirects[from_title]['redirect_to_title']

    visited_nodes = {from_title, title}
    redirects_to_another_redirect = title not in wiki_titles and title in redirects
    while redirects_to_another_redirect:
        title = redirects[title]['redirect_to_title']

        if title in visited_nodes:
            logging.warning(f'Found a redirect loop from "{from_title}"')
            return from_title  # Return the same as input

        redirects_to_another_redirect = title not in wiki_titles and title in redirects

    return title  # Does not necessarily have to be in 'wiki_titles' - e.g. special pages


def read_class_descriptions(split, dataset_config, sections):
    split_imagenet_ids = get_imagenet_ids(split, dataset_config)

    # Load matches
    matched_pages = matching.load_matches(dataset_config, expected_imagenet_ids=split_imagenet_ids)
    imagenet_ids = split_imagenet_ids.intersection(matched_pages.keys())
    logging.info(f'Found matches for {len(imagenet_ids)}/{len(split_imagenet_ids)} classes')

    # Extract wiki articles
    with open(dataset_config['meta_wiki_file'], 'rb') as f:
        meta_wiki = pickle.load(f)
    imagenet_id_details = extract_imagenet_id_details(dataset_config['imagenet_ids_to_wordnet'])

    wiki_articles_path = dataset_config['wikipedia_articles_path']
    wiki_articles = {}
    logging.info('Extracting Wikipedia articles...')
    for imagenet_id in tqdm.tqdm(imagenet_ids):
        wnid, phrases = imagenet_id_details[imagenet_id]

        class_page_texts = []
        match = matched_pages[imagenet_id]
        matched_titles = match['matched_titles']
        for wiki_title in matched_titles:
            wiki_title_id = meta_wiki[wiki_title]['id']
            article = extract_wiki_article(wiki_id=wiki_title_id,
                                           wiki_title=wiki_title,
                                           meta_wiki=meta_wiki,
                                           articles_path=wiki_articles_path)
            article_text = select_article_sections(article, sections)
            if len(article_text) == 0:
                logging.error(f'Article is empty: ({wiki_title_id}, {wiki_title} - ImageNet: {imagenet_id})')
                continue
            class_page_texts.append(article_text)

        wiki_articles[imagenet_id] = {
            'wnid': wnid,
            'phrases': phrases,
            'texts': class_page_texts,
        }

    return wiki_articles


def load_class_info(split, dataset_config):
    split_imagenet_ids = get_imagenet_ids(split, dataset_config)
    imagenet_class_details = extract_imagenet_id_details(dataset_config['imagenet_ids_to_wordnet'])
    split_wnids = {imagenet_class_details[imagenet_id][0] for imagenet_id in split_imagenet_ids}

    wordnet_structure_xml = ROOT_DIR / 'data' / 'imagenet' / 'structure_released.xml'

    tree = ET.parse(wordnet_structure_xml)
    root = tree.getroot()

    class_info = {}
    for synset in root.iter('synset'):
        wnid, words, gloss = synset.attrib['wnid'], synset.attrib['words'], synset.attrib['gloss']
        if wnid in split_wnids:
            if wnid in class_info:
                if (words, gloss) != class_info[wnid]:
                    raise RuntimeError(f'{wnid} defined in multiple different ways.')
            class_info[wnid] = words, gloss

    imagenet_class_info = {}
    for imagenet_id in split_imagenet_ids:
        wnid, phrases = imagenet_class_details[imagenet_id]
        if isinstance(phrases, str):
            phrases = [phrases]

        words, gloss = class_info[wnid]
        assert phrases == words.split(', ')

        imagenet_class_info[imagenet_id] = {
            'wnid': wnid,
            'phrases': phrases,
            'gloss': gloss
        }

    assert split_imagenet_ids == set(imagenet_class_info.keys())
    return imagenet_class_info
