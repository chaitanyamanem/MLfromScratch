
import numpy as np
import tensorflow as tf

def scaled_dot_product(Q, K, V, mask=None):
    dk = K.shape[-1] * 1.0 ## To convert the dimension integer number to flaot
    #print(f"K Shape: {K.shape}")
    attention_scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2])) / tf.sqrt(dk)
    #print(f"attention_scores shape: {attention_scores.shape}, mask shape: {mask.shape}")
    if mask is not None:
        attention_scores += mask
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attention = attention_weights @ V
    
    return attention_weights, attention

class PositionalEncoding(tf.Module):
    def __init__(self, max_seq_len, token_dim):
        self.max_seq_len = max_seq_len
        self.token_dim = token_dim
        
    def __call__(self):

        even_denom = tf.math.pow(tf.cast(10000,tf.float64),  tf.range(0, self.token_dim, 2)/ self.token_dim)
        odd_denom = tf.math.pow(tf.cast(10000,tf.float64),  tf.range(1, self.token_dim, 2)/ self.token_dim)
        position = tf.expand_dims(tf.range(0, self.max_seq_len, 1, dtype=tf.float64),-1)
        pe_even = tf.sin(position / even_denom)
        pe_odd = tf.cos(position / odd_denom)
        pe = tf.stack([pe_even, pe_odd], axis=2)
        pe = tf.reshape(pe, [self.max_seq_len, self.token_dim])
        pe = tf.cast(pe, tf.float32)
        
        return pe

class MultiheadAttention(tf.Module):
    
    def __init__(self, nheads, sequence_len, seq_dim):
        super().__init__(name=None)
        
        self.nheads = nheads
        self.sequence_len = sequence_len
        self.seq_dim = seq_dim
        self.head_dim = self.seq_dim // self.nheads
        self.qkv_net = tf.keras.layers.Dense(3 * self.seq_dim, name='qkv_net')
        self.fcnn = tf.keras.layers.Dense(self.seq_dim, name='mhfcnn')
        self.built = False
        self.attention_weights = None
        
        
        
    def __call__(self, X, mask=None):
        
        batch_size = tf.shape(X)[0]
        
        ## Create a network to generate QKV matrices
        assert self.seq_dim == X.shape[-1], "seaquence dimension given and sequence dimension in the input data is not matching "
        #print(f"X shape: {X.shape}")
        QKV = self.qkv_net(X)
        #print(f"QKV shape from qkv net: {QKV.shape}")
        QKV = tf.reshape(QKV, [batch_size, self.sequence_len]+[self.nheads, QKV.shape[-1] // self.nheads])
        QKV = tf.transpose(QKV, perm=[0,2,1,3])
        #print(f"QKV shape after heads added: {QKV.shape}")
        Q, K, V = tf.split(QKV, 3, axis=-1)
        #print(f"QKV shape individually: {(Q.shape, K.shape, V.shape)}")
        self.attention_weights, attention_embeddings = scaled_dot_product(Q, K, V, mask)
        #print(f"Attention weights and embeddings shape: {(self.attention_weights.shape, attention_embeddings.shape)}")
        #batch_size = attention_embeddings.shape[0]
        attention_embeddings = tf.reshape(attention_embeddings, shape=[batch_size,self.sequence_len,self.nheads*self.head_dim])
        #print(f"Attention embeddings shape before NN: {(self.attention_weights.shape, attention_embeddings.shape)}")
        attention_out  = self.fcnn(attention_embeddings)
        #print(f"final attention block output shape: {attention_out.shape}")
        
        return attention_out
        
        ## Break that into 
        
        
class PositionwiseFeedForward(tf.Module):

    def __init__(self, d_model, hidden, drop_prob):
        super().__init__(name=None)
        self.linear1 = tf.keras.layers.Dense(hidden, name='pffl1')
        self.linear2 = tf.keras.layers.Dense(d_model, name='pffl2')
        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(drop_prob)

    def __call__(self, x):
        x = self.linear1(x)
        #print(f"x after first linear layer: {x.shape}")
        x = self.relu(x)
        #print(f"x after activation: {x.shape}")
        x = self.dropout(x)
        #print(f"x after dropout: {x.shape}")
        x = self.linear2(x)
        #print(f"x after 2nd linear layer: {x.shape}")
        return x    
    
class EncoderBlock(tf.Module):
    def __init__(self, nheads, sequence_len, seq_dim, hidden_units, drop_prob):
        super().__init__(name=None)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(name='lnorm1')
        self.layer_norm2 = tf.keras.layers.LayerNormalization(name='lnorm2')
        self.mha = MultiheadAttention( nheads, sequence_len, seq_dim)
        self.pff = PositionwiseFeedForward(seq_dim, hidden_units, drop_prob)
        
        
    def __call__(self, X, mask=None):
        X_A = self.layer_norm1(X)
        X_A = self.mha(X_A, mask)
        X_A = X + X_A
        X_F = self.layer_norm2(X_A)
        X_F = self.pff(X_F)
        X_out = X_A + X_F
        
        ## collectign trainable variables
        """
        self.trainable_variables = self.layer_norm1.trainable_variables + \
                                        self.mha.trainable_variables + \
                                            self.layer_norm2.trainable_variables + \
                                                self.pff.trainable_variables
        """
        return X_out
        
class Encoder(tf.Module):
    def __init__(self, nblocks, nheads, sequence_len, seq_dim, hidden_units, drop_prob=0.1 ):
        self.encoder_blocks = [EncoderBlock(nheads, sequence_len, seq_dim, hidden_units, drop_prob) for _ in range(nblocks)]
        
        
    def __call__(self, X, mask=None):
        ##
        for encoder_block in self.encoder_blocks:
            X = encoder_block(X, mask)
        return X
    
class Preprocessing():
    def __init__(self, max_vocab, max_sequence_len):
        self.vectorizer = tf.keras.layers.TextVectorization(            
            max_tokens = max_vocab,
            output_mode='int',
            output_sequence_length=max_sequence_len
        )
        
    def adapt(self, X):
        self.vectorizer.adapt(X)
        
    def __call__(self, X):
        return self.vectorizer(X)
        
        
class CBERT(tf.keras.Model):
    def __init__(self, heads, endocer_layers, preprocessing_model, max_vocab, max_sequence_len, embedding_dim, hidden_units, dropout_rate):
        super().__init__(name=None)
        self.max_vocab = max_vocab
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.heads = heads        
        self.vectorizer = preprocessing_model
        self.embedding_layer = tf.keras.layers.Embedding(max_vocab, embedding_dim, mask_zero=True)
        self.pe = PositionalEncoding(max_sequence_len, embedding_dim)
        self.encoder = Encoder(
            nblocks=endocer_layers, nheads=heads, sequence_len= max_sequence_len,
            seq_dim=embedding_dim, hidden_units=hidden_units, drop_prob=dropout_rate
                         )
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.test_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name="accuracy")
       
        
    def __call__(self, X):
        
        batch_size = tf.shape(X)[0]
        embeddings = self.embedding_layer(X)
        input_mask = self.embedding_layer.compute_mask(X)

        #get positional encodings 
        positional_encodings = self.pe()

        #Now make the final input to the Encode part of the transformer
        # That is embeddings + positional embeddings

        X = embeddings + positional_encodings
        ## reshaping input mask required for the encoder
        ## in the shape [batch, heads, sequnce_len, sequnce_len]
        ## middle two dimenstio are keepign ones as matrix operation automatically broadcast
        input_mask_for_encoder = tf.where(input_mask==True, 0.0, float("-inf"))        
        input_mask_for_encoder = tf.reshape(input_mask_for_encoder, shape=[batch_size,1,1,input_mask_for_encoder.shape[-1]])
        encoder_output = self.encoder(X, input_mask_for_encoder)

        ##averaging the embeddings of all tokens in the sequence
        averaged_embeddings = tf.keras.layers.GlobalAveragePooling1D()(encoder_output)
        ## output probabilities 
        y_hat = self.output_layer(averaged_embeddings)
        
        return y_hat
        
        

        
    def train_step(self, data):        
        
        X,y = data
        #y = tf.expand_dims(y, axis=-1)
          
        #Vectorize the raw texts    
        X = self.vectorizer(X)

        #convert the word vectors into embeddings
        with tf.GradientTape() as tape:            
            y_pred = self(X)            
            loss = self.compute_loss(y=y, y_pred=y_pred)
        

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_metric.result()}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data        
        #Vectorize the raw texts    
        x = self.vectorizer(x)
        # Compute predictions
        y_pred = self(x)
        # Updates the metrics tracking the loss
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        self.test_accuracy_metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss":loss, "accuracy":self.test_accuracy_metric.result()}
        
        
        
