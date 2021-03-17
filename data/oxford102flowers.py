from pathlib import Path
import pickle
import logging

from data import read_class_names_list


def select_samples(sample_names, filter_class_names):
    logging.info(f'Selecting {len(filter_class_names)} classes out of {len(sample_names)} available')
    return {class_name: sample_names[class_name] for class_name in filter_class_names}


def read_sentences(file):
    with open(file, 'r') as f:
        return f.read().splitlines()


def read_all_sentences(data_dir, sample_names):
    sentences_data = {}
    for class_name, image_names in sample_names.items():
        for img_name in image_names:
            sents = read_sentences(data_dir / class_name / (img_name + '.txt'))
            sentences_data.update({(class_name, img_name, i): s for i, s in enumerate(sents)})
    return sentences_data


def find_all_image_names(class_dir):
    txt_files = sorted(class_dir.glob('*.txt'))
    image_names = [f.stem for f in txt_files]
    return image_names


def find_all_classes_and_images(data_dir):
    class_dirs = sorted(filter(lambda f: f.is_dir(), data_dir.glob('*')))
    assert len(class_dirs) == 102

    samples = {}
    for dir in class_dirs:
        class_name = dir.name
        samples[class_name] = find_all_image_names(dir)

    assert sum([len(d) for d in samples.values()]) == 8189

    return samples


def read_sentence_data(data_dir, split_file):
    txt_data_dir = data_dir / 'text_c10'
    sample_names = pickle.load(open(data_dir / 'class_names_to_image_names.pkl', 'rb'))

    split_class_names = read_class_names_list(split_file)
    split_sample_names = select_samples(sample_names, split_class_names)

    return read_all_sentences(txt_data_dir, split_sample_names)


def main():
    data_dir = Path('PATH/data/zsl/learning_deep_representations_of_fine_grained_visual_descriptions/cvpr2016_flowers')
    data_splits_dir = Path('PATH/data/zsl/zsl_a_comprehensive_evaluation/xlsa17/data/CUB')

    # txt_data_dir = data_dir / 'text_c10'
    # samples = find_all_classes_and_images(txt_data_dir)
    # pickle.dump(samples, open(data_dir / 'class_names_to_image_names.pkl', 'wb'))

    sentence_samples = read_sentence_data(data_dir, split_file=data_splits_dir / 'trainvalclasses.txt')
    pass


if __name__ == '__main__':
    main()
