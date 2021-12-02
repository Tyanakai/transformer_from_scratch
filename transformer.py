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
    一層のEncoder

    Attributes:
        at_weight_dim: attention機構で使用する重みの次元 
        num_heads: multi head attentionのhead数
        ffn_weight_dim: 全結合層の重みの次元。embeddingの次元に一致させる
        dropout_rate: dropout層のパラメータ
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
        """
        Args:
            input: tensor (batch_size, max_length, hidden_dim)
            attention_mask: np.array (batch_size, max_length)

        Returns:
            tensor (batch_size, max_length, hidden_dim)
        """    
        out1 = self.self_attention(input, attention_mask)
        out1 = self.layer_norm1(input + out1)

        out2 = self.ffn(out1)
        out2 = self.layer_norm2(out1 + out2)
        return out2


class Decoder(tf.keras.models.Model):
    """
    一層のDecoder

    Attributes:
        at_weight_dim: attention機構で使用する重みの次元 
        num_heads: multi head attentionのhead数
        ffn_weight_dim: 全結合層の重みの次元。embeddingの次元に一致させる
        dropout_rate: dropout層のパラメータ
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


    def call(self, 
             decoder_input, 
             decoder_attention_mask, 
             encoder_output,
             encoder_attention_mask
             ):
        """
        Args:
            decoder_input: decoder側の入力tensor (batch_size, decoder_max_length, hidden_dim)
            decoder_attention_mask: np.array (batch_size, decoder_max_length)
            encoder_output: encoder側の最終出力tensor (batch_size, encoder_max_length, hidden_dim)
            encoder_attention_mask: np.array (batch_size, encoder_max_length)
        
        Returns:
            tensor (batch_size, decoder_max_length, hidden_dim)
        """
        
        out1 = self.self_attention(decoder_input, decoder_attention_mask)
        out1 = self.layer_norm1(decoder_input + out1)
        
        out2 = self.ed_attention(
            out1, decoder_attention_mask, encoder_output, encoder_attention_mask)
        out2 = self.layer_norm2(out1 + out2)

        out3 = self.ffn(out2)
        out3 = self.layer_norm3(out2 + out3)
        return out3


class Transformer(tf.keras.models.Model):
    """
    Attributes:
        encoder_num_vocabs: encoder側の語彙数
        decoder_num_vocabs: decoder側の語彙数
        hidden_dim: embeddingベクトル及びEncoder,Decoder層の出力ベクトルの次元
        at_weight_dim: attention機構で用いる重みの次元
        num_heads: multi head attentionのhead数
        dropout_rate: dropout層のパラメータ
        num_encoders: Encoder層を積み上げる個数
        num_decoders: Decoder層を積み上げる個数
    """

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
        self.encoder_pe_layer = PositionalEncoder()
        self.decoder_pe_layer = PositionalEncoder()

        
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
        """
        Args:
            encoder_input_ids: encoder側の入力token id np.array (batch_size, encoder_max_length)
            encoder_attention_mask: np.array (batch_size, encoder_max_length)
            decoder_input: decoder側の入力token id np.array (batch_size, decoder_max_length)
            decoder_attention_mask: np.array (batch_size, decoder_max_length)
        
        Returns:
            tensor (batch_size, decoder_max_length, hidden_dim)       
        """
        encoder_vec = self.encoder_embedding_layer(encoder_input_ids)
        encoder_vec = self.encoder_pe_layer(encoder_vec)

        for encoder in self.encoders_list:
            encoder_vec = encoder(encoder_vec, encoder_attention_mask)

        decoder_vec = self.decoder_embedding_layer(decoder_input_ids)
        decoder_vec = self.decoder_pe_layer(decoder_vec)

        for decoder in self.decoders_list:
            decoder_vec = decoder(
                decoder_vec, decoder_attention_mask, encoder_vec, encoder_attention_mask)
        
        vocab_prob = self.vocab_prob_layer(decoder_vec)
        
        return {"vocab_prob": vocab_prob, "last_hidden_state": decoder_vec}