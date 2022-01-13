import os
import json
from collections import Counter
input_file = open('MSASL_test.json')
videos = json.load(input_file)

gloss_list = list()
for i in videos:
    gloss_list.append(i['text'])

# print(dict(Counter(a)).items())
print("gloss #: ",len(set(gloss_list)))
c_dict = { k:v  for k,v in dict(Counter(gloss_list)).items() if v > 10}
print("data over 100:", c_dict)
print(len(c_dict))

