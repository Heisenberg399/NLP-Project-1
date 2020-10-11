#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
import seaborn as sns


# marktwain = open(r"marktwain.txt",encoding = 'utf-8)

# In[2]:


f = open('marktwain.txt', encoding = 'utf-8')
marktwain = f.read()


# In[3]:


marktwain = re.sub(r"http\S+","", marktwain, flags = re.MULTILINE)
marktwain = re.sub(r'www\S','', marktwain)
marktwain = re.sub(r'[^(a-zA-Z)\s]','',marktwain)
marktwain = re.sub(r' (\n)',' ', marktwain)
marktwain = re.sub(r' ',' ', marktwain)
marktwain = re.sub(r'\b[A-Z]+(?:\s+[A-Z]+)*\b','',marktwain)


# In[4]:


marktwain_tockens = word_tokenize(marktwain)


# In[5]:


fdist = FreqDist(marktwain_tockens)


# In[6]:


sns.set(style = 'whitegrid')

df = pd.DataFrame(marktwain_tockens)
sns.countplot(x = df[0], order = df[0].value_counts().iloc[:40].index)
plt.xticks(rotation = 90)
plt.xlabel('Tokens')
plt.title('Freq distribution of words in marktwain.txt')
plt.savefig('freq_marktwain.jpg')
plt.show()


# In[7]:


wordcloud = WordCloud(width = 800, height = 800, 
                 background_color ='white', 
                 
                 min_font_size = 10).generate(marktwain)
plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud)
plt.axis("off") 
plt.tight_layout(pad = 0)


# In[16]:


stop_words = set(stopwords.words('english'))
stop_words_2 = set(STOPWORDS)

marktwain_tockens_sw = [w for w in marktwain_tockens if not w in 
                       stop_words and  not w in stop_words_2 ]
marktwain_noSW = ' '
marktwain_noSW = marktwain_noSW.join(marktwain_tockens_sw)


# In[17]:


wordcloud = WordCloud(width = 800, height = 800, 
                 background_color ='white', 
                  
                 min_font_size = 10).generate(marktwain_noSW)
plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud)
plt.axis("off") 
plt.tight_layout(pad = 0)


# In[19]:


fdistnew =FreqDist(marktwain_tockens_sw)
df_new = pd.DataFrame(marktwain_tockens_sw)

sns.countplot(x = df_new[0], order = df_new[0].value_counts().iloc[:40].index)
plt.xticks(rotation = 90)
plt.xlabel('Tokens')
plt.title('Freq distribution of words without StopWords in marktwain.txt')

plt.show()


# In[12]:


fdistnew


# In[13]:


fdist


# In[20]:


MT_word_len = [len(w) for w in marktwain_tockens]
fdist_len = FreqDist(MT_word_len)


# In[21]:


df_len = pd.DataFrame(MT_word_len)

sns.countplot(x = df_len[0], order = df_len[0].value_counts().iloc[:40].index)
plt.xticks(rotation = 90)
plt.xlabel('Tokens')
plt.title('Count vs Length for marktwain.txt')

plt.show()


# In[22]:


MT_tags = marktwain_noSW.split(' ')
tagged = nltk.pos_tag(MT_tags)

dic = {}
for i in tagged:
    dic[i] = dic.get(i, 0) + 1
    


# In[24]:


#making a small dictionary of only first 50 words

d = {}
d = (dict(list(dic.items())[0:50]))

keys = list(d.keys())

x = [key for i, key in keys]
values = d.values()

plt.bar(x,values)
plt.title('Freq vs Tags for marktwain.txt')
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.show()


# In[ ]:




