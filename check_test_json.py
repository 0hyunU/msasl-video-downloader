import os
import json
from collections import Counter
print(os.getcwd())
json_path=os.path.join(os.getcwd(),"MS-ASL","MS-ASL","MSASL_train.json")
input_file = open(json_path)
videos = json.load(input_file)

gloss_list = list()
for i in videos:
    gloss_list.append(i['text'])

# print(dict(Counter(a)).items())
print("gloss #: ",len(set(gloss_list)))
c_dict = { k:v  for k,v in dict(Counter(gloss_list)).items() if v > 10}
c_dict = sorted(c_dict.items(), key=lambda x: x[1])
print("data over 10:", c_dict)
print(len(c_dict))

