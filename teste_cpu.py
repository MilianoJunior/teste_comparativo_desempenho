# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:38:03 2022

@author: jrmfi
"""

import datetime
start_time = datetime.datetime.now()
# insert code snippet here
for s in range(1,1000**3):
    if (s%1000 == 0):
        print(s)
end_time = datetime.datetime.now()
print(end_time - start_time)