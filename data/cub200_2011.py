from pathlib import Path
import pickle

from data import read_class_names_list, oxford102flowers


def read_class_descriptions(data_dir, split_file):
    split_class_names = read_class_names_list(split_file)

    split_txt_descriptions = {}
    for class_name in split_class_names:
        class_file_pttrn = f'{class_name.lstrip("0")}[0-9].txt'
        matched_files = list(data_dir.glob(class_file_pttrn))
        if len(matched_files) != 1:
            raise RuntimeError(f'Could not find file: {data_dir / class_file_pttrn}')
        class_file = matched_files[0]

        with open(class_file, 'r', encoding='latin1') as f:
            txt_description = list(filter(lambda x: len(x) > 0, f.read().splitlines()))
            split_txt_descriptions[class_name] = txt_description

    return split_txt_descriptions


def read_sentence_data(data_dir, split_file):
    return oxford102flowers.read_sentence_data(data_dir, split_file)


def find_all_classes_and_images(data_dir):
    class_dirs = sorted(filter(lambda f: f.is_dir(), data_dir.glob('*')))
    assert len(class_dirs) == 200

    samples = {}
    for dir in class_dirs:
        class_name = dir.name
        samples[class_name] = oxford102flowers.find_all_image_names(dir)

    assert sum([len(d) for d in samples.values()]) == 11788

    return samples


def main():
    data_dir = Path('PATH/data/zsl/learning_deep_representations_of_fine_grained_visual_descriptions/cvpr2016_cub')
    data_splits_dir = Path('PATH/data/zsl/zsl_a_comprehensive_evaluation/xlsa17/data/CUB')

    # txt_data_dir = data_dir / 'text_c10'
    # samples = find_all_classes_and_images(txt_data_dir)
    # pickle.dump(samples, open(data_dir / 'class_names_to_image_names.pkl', 'wb'))

    sentence_data = read_sentence_data(data_dir, split_file=data_splits_dir / 'trainvalclasses.txt')
    pass

    # split_file = data_splits_dir / 'trainvalclasses.txt'
    # data_dir = Path('PATH/data/zsl/write_a_classifier/1202-Elhoseiny-sup/BirdsText/')
    # read_class_descriptions(data_dir, split_file)


if __name__ == '__main__':
    main()
