"""
transformerを構成するencoder, decoderと
transformerの全体
"""

import tensorflow as tf

from layers import (
    EncoderSelfAttention, 
    DecoderSelfAttention,
    EncoderDecoderAttention,
    LayerNormalizer,
    FeedForwardNeuralBlock,
    PositionalEncoder
)

class Encoder(tf.keras.models.Model):
    """
    input : (batch_size, max_length, embed_dim)
    attention_mask : (batch_size, max_length)
    """ 

    def __init__(
        self, 
        at_weight_dim=512, 
        num_heads=8,
        ffn_weight_dim=256, 
        dropout_rate=0.2,
        **kwargs
        ):

        super().__init__(**kwargs) 
        self.at_weight_dim = at_weight_dim
        self.num_heads = num_heads
        self.ffn_weight_dim = ffn_weight_dim
        self.dropout_rate = dropout_rate

        self.self_attention = EncoderSelfAttention(
            self.at_weight_dim, self.num_heads)
        self.layer_norm1 = LayerNormalizer()
        self.layer_norm2 = LayerNormalizer()
        self.ffn = FeedForwardNeuralBlock(self.ffn_weight_dim, self.dropout_rate)

        
    def call(self, input, attention_mask):    
        out1 = self.self_attention(input, attention_mask)
        out1 = self.layer_norm1(input + out1)

        out2 = self.ffn(out1)
        out2 = self.layer_norm2(out1 + out2)
        return out2


class Decoder(tf.keras.models.Model):
    """
    input : (batch_size, max_length, embed_dim)
    attention_mask : (batch_size, max_length)
    encoder_output : (batch_size, mas_length, embed_dim)
    """

    def __init__(
        self,  
        at_weight_dim=512, 
        num_heads=8,
        ffn_weight_dim=256,
        dropout_rate=0.2,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.at_weight_dim = at_weight_dim
        self.num_heads = num_heads
        self.ffn_weight_dim = ffn_weight_dim
        self.dropout_rate = dropout_rate

        self.self_attention = DecoderSelfAttention(
            self.at_weight_dim, self.num_heads)
        self.ed_attention = EncoderDecoderAttention(
            self.at_weight_dim, self.num_heads)
        self.ffn = FeedForwardNeuralBlock(self.ffn_weight_dim, self.dropout_rate)
        self.layer_norm1 = LayerNormalizer()
        self.layer_norm2 = LayerNormalizer()
        self.layer_norm3 = LayerNormalizer()


    def call(self, input, attention_mask, encoder_output):
        out1 = self.self_attention(input, attention_mask)
        out1 = self.layer_norm1(input + out1)
        
        out2 = self.ed_attention(out1, attention_mask, encoder_output)
        out2 = self.layer_norm2(out1 + out2)

        out3 = self.ffn(out2)
        out3 = self.layer_norm3(out2 + out3)
        return out3


class Transformer(tf.keras.models.Model):

    def __init__(self,
                 encoder_num_vocabs,
                 decoder_num_vocabs,
                 hidden_dim=256,
                 at_weight_dim=512, 
                 num_heads=8,
                 dropout_rate=0.2,
                 num_encoders=8,
                 num_decoders=8,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        self.encoder_num_vocabs = encoder_num_vocabs
        self.decoder_num_vocabs = decoder_num_vocabs
        self.hidden_dim = hidden_dim
        self.at_weight_dim = at_weight_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders

        self.encoder_embedding_layer = tf.keras.layers.Embedding(
            encoder_num_vocabs, hidden_dim)
        self.decoder_embedding_layer = tf.keras.layers.Embedding(
            decoder_num_vocabs, hidden_dim)
        self.pe_layer = PositionalEncoder()
        
        self.encoders_list = []
        self.decoders_list = []

        for _ in range(self.num_encoders):
            self.encoders_list.append(
                Encoder(at_weight_dim=at_weight_dim,
                        num_heads=num_heads,
                        ffn_weight_dim=hidden_dim,
                        dropout_rate=dropout_rate)
                )
        for _ in range(self.num_decoders):
            self.decoders_list.append(
                Decoder(at_weight_dim=at_weight_dim,
                        num_heads=num_heads,
                        ffn_weight_dim=hidden_dim,
                        dropout_rate=dropout_rate)
                )
            
        self.vocab_prob_layer = tf.keras.layers.Dense(
            decoder_num_vocabs, name="vocab_prob_layer", activation="softmax")
        
            
    def call(self, 
             encoder_input_ids, 
             encoder_attention_mask,
             decoder_input_ids,
             decoder_attention_mask
             ):
        encoder_vec = self.encoder_embedding_layer(encoder_input_ids)
        encoder_vec = self.pe_layer(encoder_vec)

        for encoder in self.encoders_list:
            encoder_vec = encoder(encoder_vec, encoder_attention_mask)

        decoder_vec = self.decoder_embedding_layer(decoder_input_ids)
        decoder_vec = self.pe_layer(decoder_vec)

        for decoder in self.decoders_list:
            decoder_vec = decoder(
                decoder_vec, decoder_attention_mask, encoder_vec)
        
        vocab_prob = self.vocab_prob_layer(decoder_vec)
        
        return {"vocab_prob": vocab_prob, "last_hidden_state": decoder_vec}
    