import unittest
import numpy as np

from text_encoders import AlbertEncoder


class TestAlbertTextEncoder(unittest.TestCase):

    def setUp(self):
        self.albert = AlbertEncoder({
            'model_name': 'albert-base-v2',
            'summary_extraction_mode': 'sum_tokens',
            'aggregate_long_text_splits_method': 'mean',
            'aggregate_descriptions_method': 'sum_representations',
            'overlap_window': 5,
            'max_length': 20,
        })
        self.texts = [
            'Very short',
            'A long text. ' * 20,
            'Even longer text than the previous. ' * 100
        ]

    def test_batch_2_is_same_as_batch_1(self):
        batch_1_emb = []
        for t in self.texts:
            t_emb = self.albert.encode_multiple_descriptions([t])
            batch_1_emb.append(t_emb)

        batch_1_emb = np.sum(batch_1_emb, axis=0)

        batch_2_emb = self.albert.encode_multiple_descriptions(self.texts)

        np.testing.assert_almost_equal(batch_2_emb, batch_1_emb)


if __name__ == '__main__':
    unittest.main()
