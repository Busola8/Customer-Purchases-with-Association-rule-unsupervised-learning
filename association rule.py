# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:15:03 2024

@author: busola
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


data = pd.read_csv("Market_Basket_Optimisation.csv")
data
data.info()


Transactions = []

for row in range(0, 7499):
    each_transaction = []
    for column in range(0, 20):
        item = data.iloc[row, column]
        if not isinstance(item, float):
            each_transaction.append(item)
    Transactions.append(each_transaction) 
    
    
Encoder = TransactionEncoder()
model = Encoder.fit_transform(Transactions)
df = pd.DataFrame(model, columns = Encoder.columns_)

support = apriori(df, min_support = 0.03, use_colnames = True, low_memory= True)
confidence = association_rules(support, metric = "confidence", min_threshold = 0.03)
lift = association_rules(support,metric = "lift", min_threshold = 1.1)


