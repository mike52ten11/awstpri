#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[5]:


a = np.array([
    [1,2,3],
    [4,5,6]
], dtype=np.float32)


# In[8]:


print(a.ndim)#數據維度


# In[9]:


print(a.shape)#數據形狀(列x行)


# In[ ]:


print(a.dtype)#數據型態


# In[10]:


print(np.average(a))#平均


# In[12]:


a.reshape(-1)#變成1維


# In[13]:


a.reshape((3,2))#變成3x2


# In[15]:


a.T#a轉置


# In[20]:


a.transpose((1,0))#a 列轉置


# In[21]:


a.transpose((0,1))#a 行轉置


# In[ ]:




