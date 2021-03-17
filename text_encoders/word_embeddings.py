import re
import tqdm
import gensim
import stanza
import logging
import numpy as np
from pathlib import Path


def load_embeddings(emb_file):
    if re.match(r".*\.txt$", str(emb_file)):
        return load_txt_embeddings(emb_file)
    elif re.match(r".*\.bin$", str(emb_file)):
        return load_bin_embeddings(emb_file)
    else:
        raise ValueError(f'Do not know how to read the file: "{emb_file}"')


def load_bin_embeddings(emb_file):
    m = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=True)
    return m, m.vector_size


def load_txt_embeddings(emb_file):
    embeddings = {}
    n_expected_tokens = None
    dim = None
    logging.info(f'Loading embeddings "{emb_file}"')
    with open(emb_file, 'r') as f:

        if re.match(r"^enwiki_.*d\.txt", str(emb_file)):
            n_expected_tokens, dim = (float(x) for x in f.readline().rstrip())

        for line in tqdm.tqdm(f, unit='tokens'):
            if not line.strip():
                continue
            token, emb_str = line.split(' ', 1)
            emb = [float(x) for x in emb_str.split(' ')]

            if dim is None:
                dim = len(emb)
            else:
                assert len(emb) == dim

            embeddings[token] = np.array(emb)

    if n_expected_tokens:
        assert len(embeddings) == n_expected_tokens

    logging.info(f'Loaded: {len(embeddings)} embedding tokens.')
    return embeddings, dim


class WordEmbeddings:

    def __init__(self, emb_config, env_config_emb):
        self.emb_config = emb_config
        self.env_config_emb = env_config_emb
        self.emb_file = Path(self.env_config_emb.embeddings_dir) / self.emb_config.embeddings_file
        self.emb, self.emb_dim = load_embeddings(self.emb_file)

        # stanza.download('en')
        self.tokenizer = stanza.Pipeline('en', processors='tokenize')

    def tokenize(self, text):
        doc = self.tokenizer(text)
        for s in doc.sentences:
            for token in s.tokens:
                yield token.text

    def encode_long_text(self, text):
        if self.emb_config.lowercase:
            text = text.lower()

        # Compute moving average
        i = 0
        t = 0
        mean_feats = np.zeros(self.emb_dim)
        for token in self.tokenize(text):
            if token in self.emb:
                mean_feats += (self.emb[token] - mean_feats) / (i+1)
                i += 1
            t += 1

        if i < 1:
            raise RuntimeError(f'Could not embed any tokens')
        logging.debug(f'Encoded tokens: {i}/{t}={i/t}')

        return mean_feats

    def encode_multiple_descriptions(self, long_texts_list):
        assert isinstance(long_texts_list, list)
        agg_method = self.emb_config.aggregate_descriptions_method

        feats = [self.encode_long_text(t) for t in long_texts_list]
        if agg_method == 'mean_representations':
            return np.mean(feats, axis=0)
        elif agg_method == 'sum_representations':
            return np.sum(feats, axis=0)

        raise ValueError(f'Cannot recognize aggregation method for multiple descriptions: {agg_method}')
