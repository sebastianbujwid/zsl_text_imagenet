from pathlib import Path

from data import oxford102flowers, cub200_2011, imagenet


def load_img_descriptions(dataset_name, split_filename, env_config_datasets):
    dataset_config = env_config_datasets[dataset_name]
    data_dir = Path(dataset_config['img_text_descriptions_data_path'])
    split_file = Path(dataset_config['data_splits_path'] / split_filename)

    if dataset_name == 'Oxford102-Flowers':
        return oxford102flowers.read_sentence_data(data_dir, split_file)

    if dataset_name == 'CUB-200-2011':
        return cub200_2011.read_sentence_data(data_dir, split_file)


def load_class_descriptions(dataset_name, split, env_config_datasets, sections):
    dataset_config = env_config_datasets[dataset_name]

    if dataset_name == 'ImageNet':
        return imagenet.read_class_descriptions(split, dataset_config, sections)

    split_file = Path(dataset_config['data_splits_path']) / split
    data_dir = Path(dataset_config['class_text_descriptions_data_path'])

    if dataset_name == 'CUB-200-2011':
        return cub200_2011.read_class_descriptions(data_dir, split_file)
