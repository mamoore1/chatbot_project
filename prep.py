# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:47:36 2020

@author: Mike
"""

from preprocessing import pairs
import re
import numpy as np

input_docs = list()
target_docs = list()

input_tokens = set()
target_tokens = set()

for line in pairs:
    # First element is statement, second is response
    input_doc = line[0]
    target_doc = line[1]
    # Append the statement to the input docs
    input_docs.append(input_doc)
    # Separate words from punctuation
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    # Attaching "<START>" and "<END>" to target doc
    target_doc = "<START> " + target_doc + " <END>"
    target_docs.append(target_doc)
    
    # Assigning tokens to our vocabulary set
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            # print(token)
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            # print(token)
            target_tokens.add(token)


input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))            

num_encoder_tokens = len(input_tokens)
num_target_tokens = len(target_tokens)


# Determining max sequence lengths
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

    # Making input and target feature dictionaries
input_features_dict = dict(
                        [(token, i) for i, token in enumerate(input_tokens)]
                        )
target_features_dict = dict(
                        list((token, i) for i, token in enumerate(target_tokens))
                        )

    # Making reverse features dictionaries
reverse_input_features_dict = dict(
                                (i, token) for token, i in input_features_dict.items()
                                )
reverse_target_features_dict = dict(
                                (i, token) for token, i in target_features_dict.items()
                                )
    
    # Making matrices for one hot vectors for each sentence at each timestep

encoder_input_data = np.zeros([len(input_docs), max_encoder_seq_length, num_encoder_tokens], dtype='float32')
decoder_input_data = np.zeros([len(target_docs), max_decoder_seq_length, num_target_tokens], dtype='float32')
decoder_target_data = np.zeros([len(target_docs), max_decoder_seq_length, num_target_tokens], dtype='float32')

# Making one hot vectors

for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        encoder_input_data[line, timestep, input_features_dict[token]] = 1
    
    for timestep, token in enumerate(target_doc.split()):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1
        
        if timestep > 0:
            decoder_target_data[line, timestep-1, target_features_dict[token]] = 1
            
