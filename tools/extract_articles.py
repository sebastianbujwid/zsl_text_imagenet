import tqdm
import pickle
import logging
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import matching
from definitions import ROOT_DIR
from data import imagenet


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matching_csv', required=True, type=Path)
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'configs' / 'env_config.yml')
    return parser.parse_args()


def main(args):
    output_file = args.matching_csv.parent / (args.matching_csv.stem + '_articles.pkl')

    if output_file.exists():
        raise ValueError(f'The output file "{output_file}" already exists!')

    config = OmegaConf.load(str(args.env_config))
    config.env.datasets.ImageNet.manual_matches_csv = str(args.matching_csv)
    OmegaConf.set_readonly(config, True)
    OmegaConf.set_struct(config, True)
    dataset_config = config.env.datasets.ImageNet

    imagenet_id_details = imagenet.extract_imagenet_id_details(dataset_config['imagenet_ids_to_wordnet'])
    with open(dataset_config['meta_wiki_file'], 'rb') as f:
        meta_wiki = pickle.load(f)

    imagenet_ids_titles, _ = matching.load_manual_matches(dataset_config)

    class_articles = {}
    for imagenet_id, titles in tqdm.tqdm(imagenet_ids_titles.items()):
        wnid, phrases = imagenet_id_details[imagenet_id]

        assert isinstance(titles, list)

        articles = []
        for title in titles:
            wiki_title_id = meta_wiki[title]['id']
            article = imagenet.extract_wiki_article(
                wiki_id=wiki_title_id,
                wiki_title=title,
                meta_wiki=meta_wiki,
                articles_path=dataset_config['wikipedia_articles_path']
            )
            articles.append(article)

        class_articles[imagenet_id] = {
            'wnid': wnid,
            'phrases': phrases,
            'articles': articles,
        }

    logging.info(f'Saving the output to: "{output_file}"')
    pickle.dump(class_articles, open(output_file, 'wb'))


if __name__ == '__main__':
    main(parse_args())
