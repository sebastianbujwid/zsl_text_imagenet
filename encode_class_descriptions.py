import argparse
import tqdm
import pickle
import logging
from pathlib import Path

from utils import SimpleEnv
from definitions import ROOT_DIR
from data import load_class_descriptions
from text_encoders import AlbertEncoder, WordEmbeddings

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(module)20.20s.%(funcName)20.20s -:- %(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        choices=[
                            'Oxford102-Flowers',
                            'CUB-200-2011',
                            'ImageNet',
                        ])
    parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--config', type=Path, default=ROOT_DIR / 'configs' / 'text_encoder_config.yml')
    parser.add_argument('--env_config', type=Path, default=ROOT_DIR / 'configs' / 'env_config.yml')
    parser.add_argument('--output_dir', type=Path, default=Path.cwd())
    parser.add_argument('--text_encoder', type=str, default='ALBERT',
                        choices=['ALBERT', 'WordEmbeddings'])
    parser.add_argument('--overwrite', nargs='+')
    return parser.parse_args()


def create_text_encoder(text_encoder_name, config, env_config):
    if text_encoder_name == 'ALBERT':
        return AlbertEncoder(config[text_encoder_name])

    elif text_encoder_name == 'WordEmbeddings':
        return WordEmbeddings(config[text_encoder_name], env_config.WordEmbeddings)


def encode_class_from_multiple_descriptions(encoder, data_samples):
    data_feats = {}
    for imagenet_id, vals in tqdm.tqdm(data_samples.items(), unit='classes'):
        feats = encoder.encode_multiple_descriptions(vals['texts'])
        data_feats[imagenet_id] = {k: v for k, v in vals.items() if k != 'texts'}
        data_feats[imagenet_id]['feats'] = feats
    return data_feats


def encode_class_descriptions(encoder, data_samples):
    keys, line_lists = zip(*(data_samples.items()))
    descriptions = ['\n'.join(line) for line in line_lists]

    feats = []
    for description in tqdm.tqdm(descriptions):
        feats.append(encoder.encode_long_text(description))

    data_feats = dict(zip(keys, feats))
    return data_feats


def main(args):
    dataset_name = args.dataset
    exp_env = SimpleEnv(args, configs=[args.config, args.env_config], output_dir=args.output_dir / dataset_name,
                        overwrite_config=args.overwrite)
    config = exp_env.config
    env_config = config.env

    text_encoder = create_text_encoder(args.text_encoder, config, env_config)
    description_samples = load_class_descriptions(dataset_name, args.split, env_config['datasets'],
                                                  sections=config['Wikipedia']['article_sections'])

    text_encoder_str = str(args.text_encoder)
    if text_encoder_str == 'WordEmbeddings':
        text_encoder_str += '_Lo' if config.WordEmbeddings.lowercase else '_Up'
        text_encoder_str += f'_{Path(config.WordEmbeddings.embeddings_file).with_suffix("")}'
    output_pickle_file = exp_env.output_dir \
                         / f'{text_encoder_str}_{dataset_name}_{Path(args.split).stem}_classes.pkl'

    if output_pickle_file.exists():
        raise ValueError(f'File {output_pickle_file} already exists')

    logging.info(f'Encoding dataset: {dataset_name}')
    if dataset_name == 'ImageNet':
        data_feats = encode_class_from_multiple_descriptions(text_encoder, description_samples)
    elif dataset_name in ['Oxford102-Flowers', 'CUB-200-2011']:
        data_feats = encode_class_descriptions(text_encoder, description_samples)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    logging.info(f'Saving encoded class descriptions to: {output_pickle_file}')
    pickle.dump(data_feats, open(output_pickle_file, 'wb'))


if __name__ == '__main__':
    main(parse_args())
