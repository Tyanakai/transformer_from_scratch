import unittest

import numpy as np
import tensorflow as tf

from transformer import Encoder, Decoder, Transformer


class TestEncoder(unittest.TestCase):

    def test_call(self):
        encoder = Encoder()

        test_input = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)
        attention_mask = np.zeros([2, 15], dtype=np.float32)
        attention_mask[:,:10] = 1

        self.assertEqual(encoder(test_input, attention_mask).shape, [2, 15, 256])


class TestDecoder(unittest.TestCase):

    def test_call(self):
        decoder = Decoder()

        enc_output = tf.constant(np.random.rand(2, 20, 256), dtype=tf.float32)
        dec_input = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)

        enc_mask = np.zeros([2, 20], dtype=np.float32)
        enc_mask[:,:12] = 1
        dec_mask = np.zeros([2, 15], dtype=np.float32)
        dec_mask[:,:10] = 1

        self.assertEqual(decoder(dec_input, dec_mask, enc_output, enc_mask).shape, [2, 15, 256])

    
class TestTransformer(unittest.TestCase):

    def test_call(self):
        transformer = Transformer(
            encoder_num_vocabs=20,
            decoder_num_vocabs=20,
            hidden_dim=32,
        )

        enc_input_ids = tf.constant(np.random.randint(1, 20, size=(2, 15)), dtype=tf.float32)
        dec_input_ids = tf.constant(np.random.randint(1, 20, size=(2, 18)), dtype=tf.float32)
        enc_mask = np.zeros([2, 15], dtype=np.float32)
        enc_mask[:,:10] = 1
        dec_mask = np.zeros([2, 18], dtype=np.float32)
        dec_mask[:,:12] = 1

        output = transformer(enc_input_ids, enc_mask, dec_input_ids, dec_mask)

        self.assertEqual(output["vocab_prob"].shape, [2, 18, 20])
        self.assertEqual(output["last_hidden_state"].shape, [2, 18, 32])