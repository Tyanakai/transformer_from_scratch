import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from layers import (
    EncoderSelfAttention, 
    DecoderSelfAttention,
    EncoderDecoderAttention,
    LayerNormalizer,
    FeedForwardNeuralBlock,
    PositionalEncoder
)

class TestEncoderSelfAttention(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.layer = EncoderSelfAttention(weight_dim=128, num_heads=4)

        cls.test_input = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)
        cls.attention_mask = np.zeros([2, 15], dtype=np.float32)
        cls.attention_mask[:,:10] = 1


    def test_split_transpose(self):
        self.assertEqual(self.layer.split_transpose(self.test_input).shape, [2,4,15,64])
    

    def test_create_mask_for_pad(self):
        mask_for_pad = self.layer.create_mask_for_pad(self.attention_mask)
        self.assertEqual(mask_for_pad.shape, [2,4,15,15])
        self.assertEqual(mask_for_pad.numpy()[0][0].sum(), 15 * 15 - 10 * 10)
    

    def test_call(self):
        self.assertEqual(self.layer(self.test_input, self.attention_mask).shape, self.test_input.shape)

    
class TestDecoderSelfAttention(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.layer = DecoderSelfAttention(weight_dim=128, num_heads=4)

        cls.test_input = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)
        cls.test_logit = tf.constant(np.random.rand(2, 4, 15, 15), dtype=tf.float32)
        cls.attention_mask = np.zeros([2, 15], dtype=np.float32)
        cls.attention_mask[:,:10] = 1


    def test_create_mask_for_future_input(self):
        future_mask = self.layer.create_mask_for_future_input(self.test_logit)
        self.assertEqual(future_mask.shape, [2,4,15,15])
        self.assertEqual(future_mask[0][0].numpy().sum(), 15 * 14 / 2)


    def test_call(self):
        self.assertEqual(self.layer(self.test_input, self.attention_mask).shape, [2, 15, 256])
        

class TestEncoderDecoderAttention(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.layer = EncoderDecoderAttention(weight_dim=128, num_heads=4)

        cls.enc_output = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)
        cls.dec_input = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)
        cls.attention_mask = np.zeros([2, 15], dtype=np.float32)
        cls.attention_mask[:,:10] = 1


    def test_call(self):
        self.assertEqual(self.layer(self.dec_input, self.attention_mask, self.enc_output).shape, [2,15,256])


class TestLayerNormalizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.layer = LayerNormalizer()

        cls.test_input = tf.constant(np.random.normal(loc=1.5, scale=0.5, size=(2,15,256)), dtype=tf.float32)


    def test_call(self):
        output = self.layer(self.test_input)

        self.assertEqual(output.shape, [2, 15, 256])
        self.assertAlmostEqual(output[0].numpy().mean(), 0, places=5)
        self.assertAlmostEqual(output[0].numpy().std(), 1, places=5)


class TestFeedForwardNeuralBlock(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.layer = FeedForwardNeuralBlock(hidden_dim=256, dropout_rate=0.2)

        cls.test_input = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)


    def test_call(self):
        self.assertEqual(self.layer(self.test_input).shape, [2, 15, 256])


class TestPositionalEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.layer = PositionalEncoder()

        cls.test_input = tf.constant(np.random.rand(2, 15, 256), dtype=tf.float32)


    def test_positional_vec(self):
        self.assertEqual(self.layer.positional_vec(15, 256).shape, (1, 15, 256))

    
    def test_call(self):
        self.assertEqual(self.layer(self.test_input).shape, [2, 15, 256])

