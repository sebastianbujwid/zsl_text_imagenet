import argparse
import logging
import pickle
import tqdm
import re
import xml.etree.ElementTree as etree
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikipedia_xml', required=True, type=Path)
    return parser.parse_args()


def strip_tag_name(elem):
    t = elem.tag
    idx = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


def remove_within(text, tag):
    return re.sub(f'<{tag}>.*?</{tag}>', '', text)


def uppercase_category(s):
    before_cat_after_split = re.split('(Category:)', s, maxsplit=1)
    if len(before_cat_after_split) != 3:
        logging.error(f'Unexpected category format: {s}')
        return s
    before, cat, after = before_cat_after_split
    if len(after) > 0:
        after = after[:1].upper() + after[1:]
    return ''.join([before, cat, after])


def extract_categories(text):
    """
    According to: https://en.wikipedia.org/wiki/Help:Category#Putting_pages_into_categories
    NOTE: Does not handle transclusions - "{{pagetitle}"
    """
    if text is None:
        return []
    text = remove_within(text, 'nowiki')
    text = remove_within(text, 'includeonly')
    if text is None:
        return []

    categories = re.findall(r'\[\[Category:.*?\]\]', text)
    categories = [x.strip('[]') for x in categories]
    categories = [x.split('|', 1)[0] for x in categories]  # strip Sortkey
    categories = list(map(uppercase_category, categories))
    return categories


def extract_pages_structure(wikipedia_xml):
    """
    Based on the code from: https://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    """
    article_ns = 0
    category_ns = 14

    redirects_dict = {}
    page_categories = {}
    category_parents = {}
    category_redirects = {}

    pages_to_discard = set()

    pbar = tqdm.tqdm(unit='pages')
    for event, elem in etree.iterparse(wikipedia_xml, events=('start', 'end')):
        tname = strip_tag_name(elem)

        if event == 'start':
            if tname == 'page':
                title = None
                id = None
                redirect = None
                inrevision = False
                ns = None
                text = None
                ignore = False
            elif tname == 'revision':
                # Do not pick up on revision id's
                inrevision = True
        else:
            if tname == 'title':
                title = elem.text
            elif tname == 'id' and not inrevision:
                id = int(elem.text)
            elif tname == 'redirect':
                redirect = elem.attrib['title']
            elif tname == 'ns':
                ns = int(elem.text)
            elif tname == 'text':
                text = elem.text
            elif tname == 'page':
                is_redirect_page = redirect is not None and ns == article_ns
                is_article_page = ns == article_ns
                is_category_redirect = redirect is not None and ns == category_ns
                is_category_page = ns == category_ns

                if is_redirect_page:
                    assert title not in redirects_dict, f'Titles are not unique! Found duplicates: {id} "{title}"'
                    redirects_dict[title] = {'from_id': id, 'redirect_to_title': redirect}

                elif is_article_page:
                    article_cats = extract_categories(text)
                    if len(article_cats) < 1:
                        logging.debug(f'Found no categories for: {id}, {title}')
                    if title in page_categories:
                        logging.error(f'Title are not unique!'
                                      f' Found duplicates: ({page_categories[title]["id"]}, {id}) "{title}".'
                                      f' Will ignore all pages with this title')
                        pages_to_discard.add(title)
                    else:
                        page_categories[title] = {
                            'id': id,
                            'categories': article_cats
                        }

                elif is_category_redirect:
                    assert title not in category_redirects, f'Categories are not unique!' \
                                                            f' Found duplicates: ({category_redirects[title]}, {id})' \
                                                            f' "{title}"'
                    category_redirects[title] = {'from_id': id, 'redirect_to_title': redirect}

                elif is_category_page:
                    parent_cats = extract_categories(text)
                    if len(parent_cats) < 1:
                        logging.debug(f'Found no parent categories for: {id}, {title}')
                    assert title not in category_parents, f'Categories are not unique! Found duplicates: {id} "{title}"'
                    category_parents[title] = {
                        'id': id,
                        'categories': parent_cats
                    }

                pbar.update(1)

            elem.clear()

    for x in pages_to_discard:
        del page_categories[x]

    return redirects_dict, page_categories, category_parents, category_redirects


def main(args):
    wikipedia_xml = args.wikipedia_xml

    pages_structure_file = wikipedia_xml.parent / ('pages_structure_' + wikipedia_xml.name + '.pkl')
    if pages_structure_file.exists():
        raise ValueError(f'The file with pages structure already exists')

    logging.info(f'Parsing pages...')
    redirects, page_categories, category_parents, category_redirects = extract_pages_structure(wikipedia_xml)

    logging.info(f'Found {len(redirects)} redirects, {len(page_categories)} articles,'
                 f' {len(category_parents)} category pages, {len(category_redirects)} category redirects')

    logging.info(f'Saving pages structure to {pages_structure_file}')
    with open(pages_structure_file, 'wb') as f:
        pickle.dump({
            'redirects': redirects,
            'page_categories': page_categories,
            'category_parents': category_parents,
            'category_redirects': category_redirects,
        }, f, protocol=4)

    logging.info('Done! Exiting.')


if __name__ == '__main__':
    main(parse_args())
