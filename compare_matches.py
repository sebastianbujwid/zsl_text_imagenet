import logging
import argparse
from pathlib import Path

import matching
from data import imagenet
from definitions import ROOT_DIR
from utils import SimpleEnv


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'configs' / 'env_config.yml')
    parser.add_argument('--imagenet_split', type=str, default='trainval_classes')
    parser.add_argument('--output_dir', type=Path, default=Path.cwd())
    parser.add_argument('--overwrite', nargs='+')
    return parser.parse_args()


def compare_matches(a_matches, b_matches):
    a_keys = set(a_matches.keys())
    b_keys = set(b_matches.keys())

    different_matches = {}
    a_not_b = {}
    b_not_a = {}

    for imagenet_id in a_keys.intersection(b_keys):
        if a_matches[imagenet_id] != b_matches[imagenet_id]:
            different_matches[imagenet_id] = (a_matches[imagenet_id], b_matches[imagenet_id])

    for imagenet_id in a_keys.difference(b_keys):
        a_not_b[imagenet_id] = a_matches[imagenet_id]

    for imagenet_id in b_keys.difference(a_keys):
        b_not_a[imagenet_id] = b_matches[imagenet_id]

    return different_matches, a_not_b, b_not_a


def main(args):
    exp_env = SimpleEnv(args, output_dir=args.output_dir, configs=[args.env_config],
                        overwrite_config=args.overwrite)
    config = exp_env.config
    env_config = config.env
    dataset_config = env_config.datasets.ImageNet

    imagenet_id_details = imagenet.extract_imagenet_id_details(env_config.datasets.ImageNet.imagenet_ids_to_wordnet)

    imagenet_ids = [i for i in range(1, 1001)]  # TODO - dehardcode!

    automatic_matches = matching.load_matches(dataset_config, expected_imagenet_ids=imagenet_ids)
    the_peoples_web_matches = matching.load_correspondences_the_peoples_web(env_config,
                                                                            select_imagenet_ids=imagenet_ids)

    #different_matches, only_automatic, only_alternative = compare_matches(automatic_matches, the_peoples_web_matches)

    dijkstra_was_matches = matching.load_correspondences_dijkstra_wsa(env_config, select_imagenet_ids=imagenet_ids)
    different_matches, only_automatic, only_alternative = compare_matches(automatic_matches, dijkstra_was_matches)

    a = None
    pass


if __name__ == '__main__':
    main(parse_args())
