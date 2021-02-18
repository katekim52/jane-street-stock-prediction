import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa

train = pd.read_csv("/kaggle/input/jane-street-market-prediction/train.csv")
tags=pd.read_csv("/kaggle/input/jane-street-market-prediction/features.csv")

resps_list=[c for c in train.columns if 'resp' in c]
features_list=[c for c in train.columns if 'feature' in c]
tags_list=[c for c in tags.columns if 'tag'in c]

#resps=train[resps_list]
#features=train[features_list]

tags=tags[tags_list].astype('int')
print(tags)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b = True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads):
        
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights

def get_attn_pad_mask(seq_q, seq_k, i_pad):
    # print(seq_q.shape)
    # print(seq_k.shape)
    
    batch_size, len_q=seq_q.shape[1:]
    batch_size, len_k=seq_k.shape[1:]
    pad_attn_mask=tf.math.equal(seq_k, i_pad)
    pad_attn_mask=tf.reshape(pad_attn_mask, (batch_size, len_q, len_k))
    
    return pad_attn_mask

def get_attn_decoder_mask(seq):
    print("get_attn_decoder_mask", seq)
    subsequent_mask = tf.reshape(tf.ones_like(seq), (1 if seq.shape[0]==None else seq.shape[0], seq.shape[1], seq.shape[1]))
    print(subsequent_mask)
    subsequent_mask = tf.experimental.numpy.triu(subsequent_mask, 1) # upper triangular part of a matrix(2-D)
    return subsequent_mask
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation = 'swish'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
    
#def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate = 0.1):
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,num_layers, d_model, num_heads, dff, maximum_position_encoding, rate = 0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

#         self.pos_encoding = positional_encoding(self.maximum_position_encoding, 
#                                                 self.d_model)
#         self.embedding = tf.keras.layers.Dense(self.d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim = self.maximum_position_encoding, 
                                                 output_dim = self.d_model)
        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(self.rate)
    
    def get_config(self):
        config=super().get_config().copy()
        config.update({
            'num_layers':self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout': self.dropout,
        })
        return config
    
    def call(self, x, training, mask=None):
        seq_len=tf.shape(x)[1]
        positions=tf.range(start=0, limit=seq_len, delta=1)
        #tf.range: python의 range와 같음
        
        x+=self.pos_emb(positions)
        x=self.dropout(x)
        
        for i in range(self.num_layers):
            x=self.enc_layers[i](x, training, mask)
        
        return x
    
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

MIXED_PRECISION = False
XLA_ACCELERATE = True

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')

###decoder###

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        self_attn_outputs, self_attn_prob=self.mha(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_attn_outputs = self.dropout2(self_attn_outputs)
        
        self_attn_outputs=self.layernorm1(dec_inputs+self_attn_outputs)
        
        dec_enc_attn_outputs, dec_enc_prob=self.mha(self_attn_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_attn_outputs = self.dropout2(dec_enc_attn_outputs)
        
        dec_enc_attn_outputs = self.layernorm2(self_attn_outputs + dec_enc_attn_outputs)
        
        ffn_outputs=self.ffn(dec_enc_attn_outputs)
        ffn_outputs = self.dropout3(ffn_outputs, training = training)
        ffn_outputs=self.layernorm3(dec_enc_attn_outputs+ffn_outputs)
        
        return ffn_outputs
        

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, enc_output, rate = 0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.enc_output=enc_output

#         self.pos_encoding = positional_encoding(self.maximum_position_encoding, 
#                                                 self.d_model)
#         self.embedding = tf.keras.layers.Dense(self.d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim = self.maximum_position_encoding, 
                                                 output_dim = self.d_model)
        self.dec_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate) 
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(self.rate)
        
        
    def get_config(self):
        config=super().get_config().copy()
        config.update({
            'num_layers':self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout': self.dropout,
        })
        return config
    
    def call(self, dec_inputs):
        seq_len=tf.shape(dec_inputs)[1]
        positions=tf.range(start=0, limit=seq_len, delta=1)
        #tf.range: python의 range와 같음
        
        dec_inputs+=self.pos_emb(positions)
        dec_inputs=self.dropout(dec_inputs)
        
        dec_attn_pad_mask=get_attn_pad_mask(dec_inputs, dec_inputs, 0)
        
        print("A ", dec_inputs)
        
        dec_attn_decoder_mask=get_attn_decoder_mask(dec_inputs)
        
        dec_self_attn_mask=tf.math.greater((dec_attn_pad_mask+dec_attn_decoder_mask), 0)
        dec_enc_attn_mask=get_attn_pad_mask(dec_inputs, enc_inputs, 0)
        
        self_attn_probs, dec_enc_attn_probs = [], []
        
        for layer in self.dec_layers:
            # (bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq), (bs, n_dec_seq, n_enc_seq)
            x, self_attn_prob, dec_enc_attn_prob = layer(dec_inputs, self.enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        # (b s, n_dec_seq, d_hidn), [(bs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]S
        return x
    
batch_size = 4096 * strategy.num_replicas_in_sync
num_layers = 1
d_model = 50
num_heads = 5
dff = 64
window_size = 4
dropout_rate = 0.15
weight_decay = 0
label_smoothing = 1e-2
learning_rate = 1e-3 * strategy.num_replicas_in_sync
verbose = 1
    
# model = create_model(len(features), 1, num_layers, d_model, num_heads, dff, window_size, dropout_rate, weight_decay, label_smoothing, learning_rate)
# (len(features_list), 4, num_layers, d_model, num_heads, dff, window_size, dropout_rate, weight_decay, label_smoothing, learning_rate, 20)

def create_model(num_columns, num_labels, num_layers, d_model, num_heads, dff, window_size, dropout_rate, weight_decay, label_smoothing, learning_rate, num_Emb):
    #Embedding using tags
    inp=tf.keras.layers.Input(shape=(window_size, num_columns))
    x=tf.linalg.matmul(inp, tags)
    x=tf.keras.layers.BatchNormalization()(inp)
    x=tf.keras.layers.Dense(d_model)(x)
    
    #x=tf.keras.layers.Conv2D(num_Emb, (1,1), padding='valid')(x)
    #x=tf.keras.layers.BatchNormalization()(x)
    
    #Decoder: def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate = 0.1):
    enc_output=TransformerEncoder(num_layers, d_model, num_heads, dff, window_size, learning_rate)(x)
    # print(num_layers, d_model, num_heads, dff, window_size, enc_output, learning_rate)
    dec_output=TransformerDecoder(num_layers, d_model, num_heads, dff, window_size, enc_output, learning_rate)(x) ## 수정
    
    out = tf.keras.layers.Dense(num_labels, activation = 'sigmoid')(dec_output[:, -1, :])
    
    ## 수정
    model = tf.keras.models.Model(inputs = inp, outputs = dec_output)
    model.compile(optimizer = tfa.optimizers.AdamW(weight_decay = weight_decay, learning_rate = learning_rate),
                  loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = label_smoothing), 
                  metrics = tf.keras.metrics.AUC(name = 'AUC'), 
                 )
    
    return model

# Rolling window

# Use Tensorflow Dataset
def prepare_dataset(X, y, window_size, batch_size, mode = 'training'):
    x_ds = tf.data.Dataset.from_tensor_slices(X) # tensor를 slice함. (a,b,c)->a개의 (b,c)
    y_ds = tf.data.Dataset.from_tensor_slices(y[window_size - 1:]) 
    x_ds = x_ds.window(window_size, shift = 1, drop_remainder = True) #window들 생성
    x_ds = x_ds.flat_map(lambda window: window.batch(window_size))#data.batch(n) 데이터를 n개씩 가져와 차례대로 쌓아서 배치로 만듦.
    dataset = tf.data.Dataset.zip((x_ds, y_ds))
    
    if mode == 'training':
        buffer_size = batch_size * 8
        dataset = dataset.repeat() #repeat: 데이터가 몇번 반복될지. param이 없다면 계속 반복. 보통 계속 반복하고 epoch 값으로 조절한다.
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration = True)
        dataset = dataset.batch(batch_size)#, drop_remainder = True
        
    elif mode == 'validation':
        dataset = dataset.batch(batch_size)
        dataset = dataset.cache() #dataset을 메모리 또는 로컬 저장소에 캐시.
        
    elif mode == 'testing':
        dataset = dataset.batch(batch_size)
        
    dataset = dataset.prefetch(AUTO)
    # most data input pipeline should be end with 'prefetch'.
    # This allows later elements to be prepared while the current element is being processed.
    
    return dataset

# Use Numpy [may cause Out-of-Memory (OOM) error]
def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.squeeze(np.lib.stride_tricks.as_strided(a, shape = s, strides = strides), axis = 1)

import gc

train=pd.DataFrame(data=train)
train[features_list] = train[features_list].fillna(method = 'ffill').fillna(0)

features1 = train.loc[train['date'] < 303, features_list].values
resps1 = train.loc[train['date'] < 303, resps_list].values

features2 = train.loc[(train['date'] >= 303) & (train['date'] <= 367), features_list].values
resps2 = train.loc[(train['date'] >= 303) & (train['date'] <= 367), resps_list].values

features3 = train.loc[train['date'] > 367, features_list].values
resps3 = train.loc[train['date'] > 367, resps_list].values

rubbish = gc.collect()

with strategy.scope():
    model = create_model(len(features_list), 4, num_layers, d_model, num_heads, dff, window_size, dropout_rate, weight_decay, label_smoothing, learning_rate, 20)
model.summary()

K.clear_session()
del model
rubbish = gc.collect()

from time import time

start_time_fold = time()

ckp_path = 'JSTransformer.hdf5'

with strategy.scope():
    model = create_model(len(features_list), 4, num_layers, d_model, num_heads, dff, window_size, dropout_rate, weight_decay, label_smoothing, learning_rate, 20)
    rlr = ReduceLROnPlateau(monitor = 'val_AUC', factor = 0.1, patience = 3, verbose = verbose, 
                        min_delta = 1e-4, mode = 'max')
    ckp = ModelCheckpoint(ckp_path, monitor = 'val_AUC', verbose = 0, 
                      save_best_only = True, save_weights_only = True, mode = 'max')
    es = EarlyStopping(monitor = 'val_AUC', min_delta = 1e-4, patience = 7, mode = 'max', 
                   baseline = None, restore_best_weights = True, verbose = 0)
    history = model.fit(features1, resps1, validation_data = (features2, resps2), batch_size = batch_size,
                    epochs = 1000, callbacks = [rlr, ckp, es], verbose = verbose)
    hist = pd.DataFrame(history.history)
    print(f'[{str(datetime.timedelta(seconds = time() - start_time_fold))[0:7]}] ROC AUC:\t', hist['val_AUC'].max())

K.clear_session()
del model, features1, actions1
rubbish = gc.collect()
