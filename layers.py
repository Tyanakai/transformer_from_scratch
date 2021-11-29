"""
transformerのencoder, decoderを構成するsublayer
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class EncoderSelfAttention(tf.keras.layers.Layer):
    """
    encoder側のself attention
    """
    
    def __init__(self, weight_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.weight_dim = weight_dim
        self.num_heads = num_heads


    def split_transpose(self, x):
        """
        xをheadの数に分割し、後の積のため転置する

        x = embed_vec * weight
        input : (batch_size, max_length, weight_dim)
        return : (batch_size, num_heads, max_length, weight_dim/num_heads)
        """
        x = tf.reshape(x, [x.shape[0], x.shape[1], self.num_heads, -1])
        x = tf.transpose(x, perm=[0,2,1,3])
        return x


    def create_mask_for_pad(self, attention_mask):
        """
        paddingの位置を無視する為のmaskを作る

        #1 　(batch_size, max_length, max_length)のmaskを作り
        #2 　headの数だけrepeatし
        #3 　0,1を反転させる

        attention_mask : (batch_size, max_length)　padの位置 = 0
        return : (batch_size, num_heads, max_length, max_length) padの位置 = True
        """
        at_mask = np.array([m.reshape(-1, 1) * m for m in attention_mask])  #1
        at_mask = np.repeat(at_mask[:,None,:,:], self.num_heads, axis=1)  #2
        p_mask = 1 - at_mask  #3
        return tf.cast(p_mask, tf.bool)


    def build(self, input_shape):
        self.wq = self.add_weight(
            "wq", shape=[input_shape[-1], self.weight_dim])
        self.wk = self.add_weight(
            "wk", shape=[input_shape[-1], self.weight_dim])
        self.wv = self.add_weight(
            "wv", shape=[input_shape[-1], self.weight_dim])
        self.wo = self.add_weight(
            "wo", shape=[self.weight_dim, input_shape[-1]])
        super().build(input_shape)
        
        
    def call(self, input, attention_mask):
        q = tf.matmul(input, self.wq)
        k = tf.matmul(input, self.wk)
        v = tf.matmul(input, self.wv)

        q = self.split_transpose(q)
        k = self.split_transpose(k)
        v = self.split_transpose(v)

        p_mask = self.create_mask_for_pad(attention_mask)
        mask = tf.cast(p_mask, tf.float32)

        logit = tf.matmul(q, k, transpose_b=True)
        logit += logit.dtype.min * mask   # set pad position to "-inf"

        attention_weight = tf.nn.softmax(
            logit / tf.sqrt(tf.cast(self.weight_dim, tf.float32)))
        multi_context_vec = tf.matmul(attention_weight, v)
        
        multi_context_vec = tf.transpose(multi_context_vec, perm=[0,2,1,3])
        concat_vec = tf.reshape(
            multi_context_vec, 
            shape=[input.shape[0], input.shape[1], self.weight_dim]
            )
        encoded_vec = tf.matmul(concat_vec, self.wo)
        return encoded_vec


class DecoderSelfAttention(EncoderSelfAttention):
    """
    decoder側のself attention
    未来時刻に対するマスク適用する点が、encoder側のself attentionと異なる
    """
    
    def __init__(self, weight_dim, num_heads, **kwargs):
        super().__init__(weight_dim, num_heads, **kwargs)


    def create_mask_for_future_input(self, input):
        """
        自身より未来のinputを参照しない為のmaskを作る

        input: (batch_size, num_heads, max_length, max_length)
        右上三角行列 - 対角行列　＝　未来時刻の値が1のマスク行列 (f-mask)
        [[0, 1, 1, 1]
         [0, 0, 1, 1]
         [0, 0, 0, 1] 
         [0, 0, 0, 0]]

        """
        ones = np.ones(input.shape)

        # 右上三角行列 - 対角行列
        f_mask = tf.linalg.band_part(ones, 0, -1) \
               - tf.linalg.band_part(ones, 0, 0)
        return tf.cast(f_mask, tf.bool)
        
        
    def call(self, input, attention_mask):
        k = tf.matmul(input, self.wk)
        v = tf.matmul(input, self.wv)
        q = tf.matmul(input, self.wq)

        q = self.split_transpose(q)
        k = self.split_transpose(k)
        v = self.split_transpose(v)

        logit = tf.matmul(q, k, transpose_b=True)

        f_mask = self.create_mask_for_future_input(logit) # create future mask
        p_mask = self.create_mask_for_pad(attention_mask)
        mask = tf.cast(tf.logical_or(f_mask, p_mask), tf.float32)
        
        logit += logit.dtype.min * mask  # set future or pad position to "-inf"

        attention_weight = tf.nn.softmax(
            logit / tf.sqrt(tf.cast(self.weight_dim, tf.float32)))
        multi_context_vec = tf.matmul(attention_weight, v)
        
        multi_context_vec = tf.transpose(multi_context_vec, perm=[0,2,1,3])
        concat_vec = tf.reshape(
            multi_context_vec, 
            shape=[input.shape[0], input.shape[1], self.weight_dim]
            )
        encoded_vec = tf.matmul(concat_vec, self.wo)
        return encoded_vec


class EncoderDecoderAttention(EncoderSelfAttention):
    """
    decoder側のlayer
    decoder側のself attentionの出力と共に、encoder側の出力も参照する
    """

    def __init__(self, weight_dim, num_heads, **kwargs):
        super().__init__(weight_dim, num_heads, **kwargs)
        

    def call(self, input, attention_mask, encoder_output):
        q = tf.matmul(input, self.wq)
        k = tf.matmul(encoder_output, self.wk)
        v = tf.matmul(encoder_output, self.wv)

        q = self.split_transpose(q)
        k = self.split_transpose(k)
        v = self.split_transpose(v)

        p_mask = self.create_mask_for_pad(attention_mask)
        mask = tf.cast(p_mask, tf.float32)

        logit = tf.matmul(q, k, transpose_b=True)
        logit += logit.dtype.min * mask   # set pad position to "-inf"

        attention_weight = tf.nn.softmax(
            logit / tf.sqrt(tf.cast(self.weight_dim, tf.float32)))
        multi_context_vec = tf.matmul(attention_weight, v)
        
        multi_context_vec = tf.transpose(multi_context_vec, perm=[0,2,1,3])
        concat_vec = tf.reshape(
            multi_context_vec, 
            shape=[input.shape[0], input.shape[1], self.weight_dim]
            )
        encoded_vec = tf.matmul(concat_vec, self.wo)
        return encoded_vec


class LayerNormalizer(tf.keras.layers.Layer):
    """
    layer毎に正規化を行う
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def build(self, input_shape):
        self.scale = self.add_weight(
            "scale", initializer=tf.keras.initializers.Constant(1.))
        self.bias = self.add_weight(
            "bias", initializer=tf.keras.initializers.Constant(0.))
        super().build(input_shape)


    def call(self, input):
        mean = tf.math.reduce_mean(input, axis=[1,2])[:, tf.newaxis, tf.newaxis]
        std = tf.math.reduce_std(input, axis=[1,2])[:, tf.newaxis, tf.newaxis]
        normalized = (input - mean) / (std + K.epsilon())
        output = normalized * self.scale + self.bias
        return output


class FeedForwardNeuralBlock(tf.keras.Model):
    """
    encoder, decoder両方で使用する全結合layer
    """

    def __init__(self, hidden_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.filter_layer = tf.keras.layers.Dense(
            hidden_dim*4, activation="relu", use_bias=True, name="filter_layer")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(
            hidden_dim, use_bias=True, name="output_layer")
        
      
    def call(self, input):
        x = self.filter_layer(input)
        x = self.dropout(x)
        output = self.output_layer(x)
        return output


class PositionalEncoder(tf.keras.layers.Layer):
    """
    入力されたtokenベクトルに位置ベクトルを加算するlayer
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def positional_vec(self, pos, embd_d):
        """
        位置ベクトルを計算する
        
        pos : 文におけるtokenの位置
        embd_d : tokenベクトルの次元
        """
        pos_v = np.zeros(shape=[pos, embd_d])
        for p in range(pos):
            for i in range(embd_d):
                if i % 2 == 0:
                    pos_v[p,i] = np.sin(p / np.power(10000, (i / embd_d)))
                else:
                    pos_v[p,i] = np.cos(p / np.power(10000, ((i - 1) / embd_d)))
        return pos_v[None,...]


    def build(self, input_shape):
        pos_vec = self.positional_vec(input_shape[1], input_shape[-1])
        self.pos_vec = tf.constant(pos_vec, dtype=tf.float32)
        super().build(input_shape)
        

    def call(self, input):
        return tf.add(input, self.pos_vec)

