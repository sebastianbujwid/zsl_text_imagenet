import re
import argparse
from pathlib import Path


from matching.utils import remove_comments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_ids_to_wordnet', required=True, type=Path,
                        help='ImageNet_w2v_extra.pkl file')
    parser.add_argument('--csv_files', required=True, nargs='+')
    parser.add_argument('-o', required=True, type=Path)
    return parser.parse_args()


def main(args):
    matches = {}
    for csv_f in args.csv_files:
        with open(csv_f) as f:
            for orig_line in f:
                line = str(orig_line)
                line = remove_comments(line)
                line = line.strip()
                if len(line) == 0:
                    continue

                split = re.split(r'\s*,\s*', line)
                wnid, wiki_titles = split[0], split[1:]
                if wnid in matches:
                    raise ValueError(f'"{wnid}" present multiple times!')
                matches[wnid] = orig_line

    output_csv = args.o
    print(f'Saving the output to: {output_csv}')
    with open(output_csv, 'w') as f:
        for wnid, line in matches.items():
            f.write(line)


if __name__ == '__main__':
    main(parse_args())
