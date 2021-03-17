import json
import tqdm
import pickle
import logging
import argparse
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_wiki_json', required=True, type=Path,
                        help="'meta_xxx.json' file generated with 'extract_wikipedia_text.sh'")
    return parser.parse_args()


def readable_title(raw_title):
    return raw_title.replace('&amp;', '&')


def read_wiki_meta(meta_json_file):
    logging.info(f'Loading meta json {meta_json_file} ...')
    wiki_meta_json = []
    with open(meta_json_file, mode='r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm.tqdm(f, unit='lines')):
            line = line.strip()
            if len(line) < 1:
                continue
            try:
                json_data = json.loads(line)
                assert len(json_data) == 4
                wiki_meta_json.append(json_data)
            except:
                logging.error(f'Problem with {i}: {l}')
                raise RuntimeError(f'Problem with {i}: {l}')

    logging.info(f'Finished reading the file!')
    wiki_meta = {readable_title(d['title']): d for d in wiki_meta_json}
    for title, vals in wiki_meta.items():
        vals['id'] = int(vals['id'])
        vals['title'] = readable_title(vals['title'])
    logging.info(f'Meta json loaded, {len(wiki_meta)} articles!')
    return wiki_meta


def main(args):
    meta_json_file = args.meta_wiki_json
    meta_pkl_file = meta_json_file.with_suffix('.pkl')
    if meta_pkl_file.exists():
        raise ValueError(f'File {meta_pkl_file} already exists')

    wiki_meta = read_wiki_meta(meta_json_file)
    logging.info(f'Saving pickle {meta_pkl_file}')
    with open(meta_pkl_file, 'wb') as f:
        pickle.dump(wiki_meta, f, protocol=4)
    f.close()


if __name__ == '__main__':
    main(parse_args())
