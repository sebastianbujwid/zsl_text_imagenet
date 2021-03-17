import h5py
import pickle
import argparse
import pymatreader
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_file', required=True, type=Path)
    return parser.parse_args()


def main(args):
    output_hdf5_file = args.mat_file.with_suffix('.hdf5')
    if output_hdf5_file.exists():
        raise ValueError(f'File {output_hdf5_file} already exists!')
    output_extra_pkl = args.mat_file.parent / (args.mat_file.stem + '_extra.pkl')
    if output_extra_pkl.exists():
        raise ValueError(f'File {output_extra_pkl} already exists!')

    data_dict = pymatreader.read_mat(args.mat_file)
    print(f'Found fields: {list(data_dict.keys())}')
    h = h5py.File(output_hdf5_file, 'w')
    extra_keys = []
    for k, v in data_dict.items():
        print(k)
        try:
            h.create_dataset(k, data=v)
        except TypeError:
            extra_keys.append(k)

    if len(extra_keys) > 0:
        extra_dict = {k: data_dict[k] for k in extra_keys}
        print(f'Saving extra fields to: {output_extra_pkl}')
        pickle.dump(extra_dict, open(output_extra_pkl, 'wb'))


if __name__ == '__main__':
    main(parse_args())
