# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:48:30 2020

@author: Mike
"""

import numpy as np
import re
from prep import num_encoder_tokens, num_target_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length
from testing_model import encoder_model, decoder_model

class ChatBot():
    
    negative_responses = ['no', 'nah', 'nope', 'never']
    
    exit_commands = ['exit', 'escape', 'quit', 'stop']
    
    def start_chat(self):
        reply = input('Hello and welcome to Mike\'s chatbot, which has been trained on film dialogue.\nWould you like to start chatting?\n> ')
        if reply in self.negative_responses:
            print('Okay, talk to you later')
            return
        while not self.make_exit(reply):
            reply = input(self.generate_respose(reply) + "\n> ")

    def make_exit(self, user_input):
        for exit_command in self.exit_commands:
            if exit_command in user_input:
                print('Goodbye then!')
                return True
        return False
    
    def string_to_matrix(self, user_input):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            [1, max_encoder_seq_length, num_encoder_tokens],
            dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1
        return user_input_matrix
    
    def generate_respose(self, user_input):
        test_input = self.string_to_matrix(user_input)
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
        
        decoded_sentence = decoded_sentence.replace('<START>', '').replace('<END>', '')
        return decoded_sentence
    
chatter = ChatBot()
chatter.start_chat()