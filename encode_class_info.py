import argparse
import tqdm
import pickle
import logging
import numpy as np
from pathlib import Path

from utils import SimpleEnv
from definitions import ROOT_DIR
from data import imagenet
from encode_class_descriptions import create_text_encoder


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_info', required=True, type=str,
                        choices=[
                            'phrases',
                            'gloss',
                            'phrases_gloss_mean',
                        ])
    parser.add_argument('--dataset', required=True, type=str,
                        choices=[
                            'ImageNet',
                        ])
    parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--config', type=Path, default=ROOT_DIR / 'configs' / 'text_encoder_config.yml')
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'configs' / 'env_config.yml')
    parser.add_argument('--output_dir', type=Path, default=Path.cwd())
    parser.add_argument('--text_encoder', type=str, default='ALBERT', choices=['ALBERT'])
    parser.add_argument('--overwrite', nargs='+')
    return parser.parse_args()


def main(args):
    dataset_name = args.dataset
    class_info_type = args.class_info
    exp_env = SimpleEnv(args, configs=[args.config, args.env_config], output_dir=args.output_dir / dataset_name,
                        overwrite_config=args.overwrite)
    config = exp_env.config
    env_config = config.env

    text_encoder = create_text_encoder(args.text_encoder, config, env_config)
    class_info = imagenet.load_class_info(args.split, env_config['datasets']['ImageNet'])
    output_pickle_file = exp_env.output_dir \
                         / f'{args.class_info}_{args.text_encoder}_{dataset_name}_{Path(args.split).stem}_classes.pkl'

    if output_pickle_file.exists():
        raise ValueError(f'File {output_pickle_file} already exists')

    logging.info(f'Encoding dataset: {dataset_name}')

    data_feats = {}
    for imagenet_id, vals in tqdm.tqdm(class_info.items(), unit='classes'):

        if class_info_type == 'phrases':
            feats = text_encoder.encode_multiple_descriptions(vals['phrases'])
        elif class_info_type == 'gloss':
            feats = text_encoder.encode_multiple_descriptions([vals['gloss']])
        elif class_info_type == 'phrases_gloss_mean':
            phrases_feats = text_encoder.encode_multiple_descriptions(vals['phrases'])
            gloss_feats = text_encoder.encode_multiple_descriptions([vals['gloss']])
            feats = np.mean([phrases_feats, gloss_feats], axis=0)
        else:
            raise ValueError(f'class_info_type: "{class_info_type}" not recognized!')

        data_feats[imagenet_id] = dict(vals)
        data_feats[imagenet_id]['feats'] = feats

    logging.info(f'Saving encoded class descriptions to: {output_pickle_file}')
    pickle.dump(data_feats, open(output_pickle_file, 'wb'))


if __name__ == '__main__':
    main(parse_args())
