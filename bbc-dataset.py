import os
import pandas as pd
import re

folders = ["business","entertainment","politics","sport","tech"]

#print(os.getcwd())
path = os.path.dirname(os.path.abspath(__file__))


paths = []
for i in folders:
    paths.append(path+'\\bbc'+'\\'+i)


texts = []
labels = []
labels_to_category = []
for i in range(len(paths)):
    for j in os.listdir(paths[i]):
        #  encoding = 'unicode_escape'
        with open(paths[i]+ "\\" + j, "rt",  encoding = 'unicode_escape') as f:
            temp = f.read()
            temp = temp.replace("\n"," ").replace('\r','')
            texts.append(temp)
            f.close()
        labels.append(folders[i])
        labels_to_category.append(i)
       

data = pd.DataFrame({'texts':texts,'labels':labels,'CAT':labels_to_category})
data.to_csv(path + '\\BBC_news.csv')

test = pd.read_csv('BBC_news.csv',encoding = 'latin1')