# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:44:36 2020

@author: Mike
"""
from itertools import zip_longest
import re

datapath =  "movie_lines.txt"

with open(datapath, "r", encoding='iso-8859-1') as f:
    raw_lines = f.read().split('\n')

# reversing raw lines to put them in the correct order
raw_lines.reverse()

# reducing the number of raw lines to an amount that can be turned into one hot matrices without running out of memory
raw_lines = raw_lines[:(len(raw_lines)//200)]

lines = list()

# Removing the other information from each line

for line in raw_lines:
    line = line.split('+++$+++')
    lines.append(line[-1].strip(' '))
    
# Making a paired list of lines and responses

def grouper(iterable, n, fillvalue=''):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
close_pairs = list(grouper(lines[1:], 2))

pairs = list()


# removing pairs containing racial slurs
for pair in close_pairs:
    if re.search("[Nn]igg\w+", pair[0]) or re.search("[Nn]igg\w+", pair[1]):
        continue
    else:
        #print(pair[1])
        pairs.append(pair)
            

# print(pairs[:15])