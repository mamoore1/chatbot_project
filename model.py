# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:56:15 2020

@author: Mike
"""

from prep import num_encoder_tokens, num_target_tokens, encoder_input_data,\
 decoder_input_data, decoder_target_data

from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from keras.models import Model

batch_size = 1
epochs = 1

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(256, return_state = True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]



decoder_inputs = Input(shape=(None, num_target_tokens))
decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_target_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

