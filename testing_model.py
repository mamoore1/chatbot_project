# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:46:19 2020

@author: Mike
"""
#from training_model import decoder_inputs, decoder_lstm, decoder_dense
from prep import target_features_dict, reverse_target_features_dict, max_decoder_seq_length,  num_target_tokens, num_encoder_tokens
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model, load_model
import numpy as np

dimensionality = 256
decoder_inputs = Input(shape=(None, num_target_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_dense = Dense(num_target_tokens, activation='softmax')

# importing training model

training_model = load_model('training_model.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

# building encoder test model

encoder_model = Model(encoder_inputs, encoder_states)

# building decoder input layers
latent_dim = 256

decoder_state_input_hidden = Input(shape=(latent_dim))
decoder_state_input_cell = Input(shape=(latent_dim))

decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# building decoder LSTM layer
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, \
                                                         initial_state=decoder_states_inputs)

decoder_states = [state_hidden, state_cell]

# Passing decoder_outputs through dense layer

decoder_outputs = decoder_dense(decoder_outputs)

# Building decoder test model
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(test_input):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(test_input)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1, num_target_tokens))
  # Populate the first token of target sequence with the start token.
  target_seq[0, 0, target_features_dict['<START>']] = 1.

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  decoded_sentence = ''

  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible 
    # output tokens (with probabilities) & states
    output_tokens, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = reverse_target_features_dict[sampled_token_index]
    decoded_sentence += " " + sampled_token

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, num_target_tokens))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    states_value = [hidden_state, cell_state]

  return decoded_sentence