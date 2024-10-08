#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:24:43 2024

@author: ciciwxp
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv('/Users/ciciwxp/Downloads/AC215Data - Kathy.csv')
df2 = pd.read_csv('/Users/ciciwxp/Downloads/AC215Data - Cici (1).csv').iloc[:,:6]
df3 = pd.read_csv('/Users/ciciwxp/Downloads/AC215Data - Jessie.csv')
df4 = pd.read_csv('/Users/ciciwxp/Downloads/AC215Data - Winnie.csv').iloc[:,:6]
full_tb = pd.concat([df1, df2, df3, df4], axis=0)

full_tb.replace(r'\n', ' ', regex=True, inplace=True)
full_tb.replace(r'\r\n', ' ', regex=True, inplace=True)
full_tb.rename(columns=({'number': 'id'}), inplace = True)
full_tb.to_csv('/Users/ciciwxp/Desktop/AC215/data_oct8.csv', index=False)
full_tb.to_json('//Users/ciciwxp/Desktop/AC215/data_oct8.json', orient='records', lines=True)
