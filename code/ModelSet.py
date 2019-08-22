import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, ZeroPadding1D, Conv1D, Dense, TimeDistributed, concatenate, Flatten
from keras.layers import AveragePooling1D
from keras_contrib.layers import CRF
import keras.backend as K
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        

class NERModel:
    def __init__(self, maxlen, word_dict_size, word_vec_size, class_label_count):
        self.maxlen = maxlen
        self.word_dict_size = word_dict_size
        self.word_vec_size = word_vec_size
        self.class_label_count = class_label_count
        self.model = self._build_model()
        
    def _build_model(self):
        input_layer = Input(shape=(self.maxlen,), dtype='int32', name='input_layer')
        embedding_layer = Embedding(self.word_dict_size, self.word_vec_size, name='embedding_layer')(input_layer)
        bilstm = Bidirectional(LSTM(32, return_sequences=True))(embedding_layer)
        bilstm_d = Dropout(0.1)(bilstm)
#         half_window_size = 2
#         paddinglayer = ZeroPadding1D(padding=half_window_size)(embedding_layer)

#         conv = Conv1D(nb_filter=50, filter_length=(2 * half_window_size + 1), border_mode='valid')(paddinglayer)
#         conv_d = Dropout(0.1)(conv)
#         dense_conv = TimeDistributed(Dense(50))(conv_d)
#         rnn_cnn_merge = concatenate([bilstm_d, dense_conv], axis=2)
#         dense = TimeDistributed(Dense(self.class_label_count))(rnn_cnn_merge)
        crf = CRF(self.class_label_count, sparse_target=True)
#         crf_output = crf(dense)
        crf_output = crf(bilstm_d)
        model = Model(input=[input_layer], output=[crf_output])
        model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
        model.summary()
    
        return model
    
    def train(self, data, label):
        checkpointer = ModelCheckpoint(filepath="../model/bilstm_1102_k205_tf130.w", verbose=0, save_best_only=True, save_weights_only=True) #save_weights_only=True
        history = LossHistory()
        earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')
        
        data = pad_sequences(data, self.maxlen, padding = 'post', truncating='post')
        label = pad_sequences(label, self.maxlen, padding = 'post', truncating='post')
        label = np.expand_dims(label,2)
        self.model.fit(data, label,
                       batch_size=32, epochs=500,#validation_data = ([x_test, seq_lens_test], y_test),
                       callbacks=[checkpointer, history,earlystop],
                       verbose=1,
                       validation_split=0.1,
                      )
        
    def predict(self, data, id2chunk):
        output_result = []
        self.model.load_weights("../model/bilstm_1102_k205_tf130.w")
        data = pad_sequences(data, self.maxlen, padding = 'post', truncating='post')
        result = self.model.predict(data)
        for i in range(len(result)):
            output_result.append([id2chunk.get(item[1]) for item in np.argwhere(result[i])])
        return output_result

class PolaritiesClassifier:
    def __init__(self, content_max_len, aspect_max_len, option_max_len, word_dict_size, word_vec_size):
        self.content_max_len = content_max_len
        self.aspect_max_len = aspect_max_len
        self.option_max_len = option_max_len
        self.word_vec_size = word_vec_size
        self.word_dict_size = word_dict_size
        self.model = self._build_model()
        
    def _build_model(self):
        content_input_layer = Input(shape=(self.content_max_len,), dtype='int32', name='content_input_layer')
        aspect_input_layer = Input(shape=(self.aspect_max_len,), dtype='int32', name='aspect_input_layer')
        option_input_layer = Input(shape=(self.option_max_len,), dtype='int32', name='option_input_layer')
        shared_embedding_layer = Embedding(self.word_dict_size, self.word_vec_size, name='embedding_layer')
        content_vec = shared_embedding_layer(content_input_layer)
        aspect_vec = shared_embedding_layer(aspect_input_layer)
        option_vec = shared_embedding_layer(option_input_layer)
        
        half_window_size = 2
        padding = ZeroPadding1D(padding=half_window_size)
        
        content_padding_layer = padding(content_vec)
        aspect_padding_layer = padding(aspect_vec)
        option_padding_layer = padding(option_vec)
        
        conv_layer = Conv1D(nb_filter=50, filter_length=(2 * half_window_size + 1), border_mode='valid')
        pooling_layer = AveragePooling1D(2, 2,name='pooling')
        content_conv = conv_layer(content_padding_layer)
        aspect_conv = conv_layer(aspect_padding_layer)
        option_conv = conv_layer(option_padding_layer)
        
        content_dense_conv = pooling_layer(content_conv)
        aspect_dense_conv = pooling_layer(aspect_conv)
        option_dense_conv = pooling_layer(option_conv)

        cnn_merge = concatenate([content_dense_conv, aspect_dense_conv, option_dense_conv], axis=1)
        dense = Flatten()(cnn_merge)
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(input=[content_input_layer, aspect_input_layer, option_input_layer], output=output)
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
    
    def train(self, content_data, aspect_data, option_data, label):
        checkpointer = ModelCheckpoint(filepath="../model/polarity.w", verbose=0, save_best_only=True, save_weights_only=True) #save_weights_only=True
        history = LossHistory()
        earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')
        
        content_data = pad_sequences(content_data, self.content_max_len, padding = 'post', truncating='post')
        aspect_data = pad_sequences(aspect_data, self.aspect_max_len, padding = 'post', truncating='post')
        option_data = pad_sequences(option_data, self.option_max_len, padding = 'post', truncating='post')
        self.model.fit([content_data, aspect_data, option_data], label,
                       batch_size=32, epochs=500,#validation_data = ([x_test, seq_lens_test], y_test),
                       callbacks=[checkpointer, history,earlystop],
                       verbose=1,
                       validation_split=0.1,
                      )
            
    def predict(self, content_data, aspect_data, option_data, id2polarities):
        output_result = []
        self.model.load_weights("../model/polarity.w")
        content_data = pad_sequences(content_data, self.content_max_len, padding = 'post', truncating='post')
        aspect_data = pad_sequences(aspect_data, self.aspect_max_len, padding = 'post', truncating='post')
        option_data = pad_sequences(option_data, self.option_max_len, padding = 'post', truncating='post')
        data = [content_data, aspect_data, option_data]
        result = self.model.predict(data)
        
        for item in result:
            if item[0] > 0.5:
                output_result.append('正面')
            else:
                output_result.append('负面')
        return output_result

class CategoriesClassifier:
    def __init__(self, content_max_len, aspect_max_len, option_max_len, categories_num, word_dict_size, word_vec_size):
        self.content_max_len = content_max_len
        self.aspect_max_len = aspect_max_len
        self.option_max_len = option_max_len
        self.categories_num = categories_num
        self.word_vec_size = word_vec_size
        self.word_dict_size = word_dict_size
        self.model = self._build_model()
        
    def _build_model(self):
        content_input_layer = Input(shape=(self.content_max_len,), dtype='int32', name='content_input_layer')
        aspect_input_layer = Input(shape=(self.aspect_max_len,), dtype='int32', name='aspect_input_layer')
        option_input_layer = Input(shape=(self.option_max_len,), dtype='int32', name='option_input_layer')
        shared_embedding_layer = Embedding(self.word_dict_size, self.word_vec_size, name='embedding_layer')
        content_vec = shared_embedding_layer(content_input_layer)
        aspect_vec = shared_embedding_layer(aspect_input_layer)
        option_vec = shared_embedding_layer(option_input_layer)
        
        half_window_size = 2
        padding = ZeroPadding1D(padding=half_window_size)
        
        content_padding_layer = padding(content_vec)
        aspect_padding_layer = padding(aspect_vec)
        option_padding_layer = padding(option_vec)
        
        conv_layer = Conv1D(nb_filter=50, filter_length=(2 * half_window_size + 1), border_mode='valid')
        pooling_layer = AveragePooling1D(2, 2,name='pooling')
        content_conv = conv_layer(content_padding_layer)
        aspect_conv = conv_layer(aspect_padding_layer)
        option_conv = conv_layer(option_padding_layer)
        
        content_dense_conv = pooling_layer(content_conv)
        aspect_dense_conv = pooling_layer(aspect_conv)
        option_dense_conv = pooling_layer(option_conv)

        cnn_merge = concatenate([content_dense_conv, aspect_dense_conv, option_dense_conv], axis=1)
        dense = Flatten()(cnn_merge)
        output = Dense(self.categories_num, activation='softmax')(dense)
        model = Model(input=[content_input_layer, aspect_input_layer, option_input_layer], output=output)
        
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.summary()
        return model
    
    def train(self, content_data, aspect_data, option_data, label):
        checkpointer = ModelCheckpoint(filepath="../model/categories.w", verbose=0, save_best_only=True, save_weights_only=True) #save_weights_only=True
        history = LossHistory()
        earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='min')
        
        content_data = pad_sequences(content_data, self.content_max_len, padding = 'post', truncating='post')
        aspect_data = pad_sequences(aspect_data, self.aspect_max_len, padding = 'post', truncating='post')
        option_data = pad_sequences(option_data, self.option_max_len, padding = 'post', truncating='post')
        encoded=to_categorical(label)
        self.model.fit([content_data, aspect_data, option_data], encoded,
                       batch_size=32, epochs=500,#validation_data = ([x_test, seq_lens_test], y_test),
                       callbacks=[checkpointer, history,earlystop],
                       verbose=1,
                       validation_split=0.1,
                      )
            
    def predict(self, content_data, aspect_data, option_data, id2categories):
        output_result = []
        self.model.load_weights("../model/categories.w")
        content_data = pad_sequences(content_data, self.content_max_len, padding = 'post', truncating='post')
        aspect_data = pad_sequences(aspect_data, self.aspect_max_len, padding = 'post', truncating='post')
        option_data = pad_sequences(option_data, self.option_max_len, padding = 'post', truncating='post')
        data = [content_data, aspect_data, option_data]
        result = self.model.predict(data)

        for i in range(len(result)):
            output_result.append(id2categories.get(str(np.argmax(result[i]))))
        return output_result