

import pandas as pd
import numpy as np

import textrank4zh

file = pd.read_csv('/home/tv/Downloads/query_result_desc.csv',)

desc = file['description']
desc_dic = list(desc.items())

import re
tags_dicc =list( map(lambda x:
    re.split('[\。|\~|\·|\;|\，|\、|\t|\s]',x), desc))
tags_dicc

