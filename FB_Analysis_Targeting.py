#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2,style='darkgrid')

get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#change display into using full screen
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[3]:


#import .csv file
data = pd.read_csv('FB_data_targeting.csv')


# # 1. Data Exploration & Cleaning

# ### 1. Have a first look at data

# In[53]:


data.head()


# In[ ]:





# In[54]:


data.shape


# ### 2. Drop Columns that are extra

# In[55]:


#We see that Reporting Starts and Reporting Ends are additional columns which we don't require. So we drop them
data.drop(['Reporting ends','Reporting starts'],axis=1, inplace = True)


# In[56]:


#look at the data again
data.head()


# In[57]:


#check rows and columns in data
data.shape


# #### So, there are 31 rows and 13 columns in the data

# ### 3. Deal with Null Values

# In[58]:


#let's look if any column has null values
data.isnull().sum()


# #### From this we can infer that some columns have Null values (basically blank). Let's look at them:
# **1. Results:** This happened when there was no conversion (Result).
# 
# **2. Result rate, Cost per result:** As both these metrics depend on Result, so these are also blank. 
# 
# This was bound to happen because not every single day and every ad got a result (conversion). **So it is safe to replace all nulls in Results and Result rate column with 0.**

# In[ ]:





# In[59]:


#Fill all blanks in Results with 0
data['Results'] = data['Results'].fillna(0)
data['Result rate'] = data['Result rate'].fillna(0)


# In[60]:


#check how many nulls are still there 
data.isnull().sum()


# #### Voila! Results & Result rate column has no nulls now. Let's see what column Results Type is all about. 

# Now we need to deal with **Cost per result**.
# The cases where CPA is Null means that there was no conversion. So ideally, in these cases the CPA should be very high (in case a conversion actually happened).

# #### So, let's leave this column as it is because we can't assign any value for records where no conversion happened.

# In[21]:


data.info()


# # 2. Feature Engineering

# Make new and better features from using the current available features

# ### 1. Make Cost per Click column

# Notice that our data does not have Cost per click; which is an important KPI.
# Let us create that metric using the formula: 
# **CPC = Cost / Clicks**

# In[61]:


data['CPC'] = data['Amount spent (INR)']/data['Clicks (all)']


# In[62]:


data.head()


# ### 2. We can divide Frequency in buckets

# In[63]:


data['Frequency'] = data['Frequency'].apply(lambda x:'1 to 2' if x<2
                                               else '2 to 3' if x>=2 and x<3 
                                               else '3 to 4' if x>=3 and x<4
                                               else '4 to 5' if x>=4 and x<5
                                               else 'More than 5')


# In[64]:


data.head()


# ### 3. Split Ad name into Ad Format and Ad Headline

# In[65]:


data['Ad_name'] = data['Ad name']


# In[66]:


data.head()


# In[67]:


data[['Ad Format','Ad Headline']] = data.Ad_name.str.split("-",expand=True)


# In[69]:


data.head()


# In[70]:


data.drop(['Ad name','Ad_name'],axis=1, inplace = True)


# In[71]:


data.head()


# In[72]:


data.info(verbose=1)


# In[ ]:





# In[73]:


data.to_csv('Clean_Data_Targeting.csv')


# ## Now our data is clean. Here are our features that we will use for analysis
# 
# - **1. Campaign Name** - Name of campaign
# - **2. Ad Set Name** - Targeting
# - **3. Result Type** - What type of conversion happened: page like, post enagement, in-facebook lead, store visit or custom conversion.
# - **4. Results** - How many conversions were achieved
# - **5. CTR** - Click Through Rate
# - **6. Result Rate** - Conversion Rate
# - **7. Amount spent** - How much money was spent on ad campaign
# - **8. Cost per result** - Average Cost required for 1 conversion
# - **9. Frequency** - On an average how many times did one user see the ad
# - **10. CPM** - Cost per 1000 impressions
# - **11. Convert Status** - Whether a conversion happened or not
# - **12. Ad Format** - Whether the ad crative is **Image/Video/Carousel**
# - **13. Ad Headline** - The headline used in ad
# 
# The variables having object written in front of them are categorical columns. While the rest are numerical.

# # 3. Relationship Visualization

# ## 1. Effect of Targeting + Ad Headline + Budget on Engagement & Conversion

# In[34]:


data['Ad set name'].value_counts()


# In[35]:


data['Ad Headline'].value_counts()


# ### Generic View

# In[45]:


# increase figure size 
plt.figure(figsize=(22, 7))

# subplot 1
plt.subplot(1, 6, 1)
sns.barplot(x='Ad set name', y='Amount spent (INR)', data=data, estimator=np.sum,ci=None)
plt.title("Total Amount Spent")
plt.xticks(rotation=90)


# subplot 2
plt.subplot(1, 6, 2)
sns.barplot(x='Ad set name', y='Clicks (all)', data=data, estimator=np.sum,ci=None)
plt.title("Total Clicks")
plt.xticks(rotation=90)

# subplot 3
plt.subplot(1, 6, 3)
sns.barplot(x='Ad set name', y='CTR (all)', data=data, estimator=np.mean,ci=None)
plt.title("CTR")
plt.xticks(rotation=90)

# subplot 4
plt.subplot(1, 6, 4)
sns.barplot(x='Ad set name', y='Results', data=data, estimator=np.sum,ci=None)
plt.title("Total Conversions")
plt.xticks(rotation=90)

# subplot 5
plt.subplot(1, 6, 5)

sns.barplot(x='Ad set name', y='Cost per result', data=data, estimator=np.mean,ci=None)
plt.title("Avg. Cost per Conversion")
plt.xticks(rotation=90)

# subplot 6
plt.subplot(1,6, 6)
sns.barplot(x='Ad set name', y='Result rate', data=data, estimator=np.mean,ci=None)
plt.title("CVR")
plt.xticks(rotation=90)


plt.tight_layout(pad=0.7)
plt.show()


# ### Let's Look at a Granular View

# In[48]:


sns.catplot(y='Ad Headline',x='Amount spent (INR)',col='Ad set name',data=data,kind='bar',aspect=1.3, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Amount Spent on Each Ad',fontsize=20)

sns.catplot(y='Ad Headline',x='Clicks (all)',col='Ad set name',data=data,kind='bar',aspect=1.3, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Clicks on Each Ad',fontsize=20)

sns.catplot(y='Ad Headline',x='CTR (all)',col='Ad set name',data=data,kind='bar',aspect=1.3, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Click Through Rate (CTR) of Each Ad',fontsize=20)

sns.catplot(y='Ad Headline',x='Results',col='Ad set name',data=data,kind='bar',aspect=1.3, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Conversions from Each Ad',fontsize=20)

sns.catplot(y='Ad Headline',x='Cost per result',col='Ad set name',data=data,kind='bar',aspect=1.3, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Cost per Conversion of Each Ad',fontsize=20)

sns.catplot(y='Ad Headline',x='Result rate',col='Ad set name',data=data,kind='bar',aspect=1.3, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Conversion Rate of Each Ad',fontsize=20)


# ## 2. Effect of Targeting + Ad Format + Budget on Engagement & Conversion

# ### Generic View

# In[47]:


# increase figure size 
plt.figure(figsize=(20, 6))

# subplot 1
plt.subplot(1, 6, 1)
sns.barplot(x='Ad Format', y='Amount spent (INR)', data=data, estimator=np.sum,ci=None)
plt.title("Total Amount Spent")
plt.xticks(rotation=90)


# subplot 2
plt.subplot(1, 6, 2)
sns.barplot(x='Ad Format', y='Clicks (all)', data=data, estimator=np.sum,ci=None)
plt.title("Total Clicks")
plt.xticks(rotation=90)

# subplot 3
plt.subplot(1, 6, 3)
sns.barplot(x='Ad Format', y='CTR (all)', data=data, estimator=np.mean,ci=None)
plt.title("CTR")
plt.xticks(rotation=90)

# subplot 4
plt.subplot(1, 6, 4)
sns.barplot(x='Ad Format', y='Results', data=data, estimator=np.sum,ci=None)
plt.title("Total Conversions")
plt.xticks(rotation=90)


# subplot 5
plt.subplot(1, 6, 5)
sns.barplot(x='Ad Format', y='Cost per result', data=data, estimator=np.mean,ci=None)
plt.title("Avg. Cost per Conversion")
plt.xticks(rotation=90)

# subplot 6
plt.subplot(1,6, 6)
sns.barplot(x='Ad Format', y='Result rate', data=data, estimator=np.mean,ci=None)
plt.title("CVR")
plt.xticks(rotation=90)


plt.tight_layout(pad=0.7)
plt.show()


# In[39]:


sns.catplot(x='Ad Format',y='Amount spent (INR)',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Amount Spent on Each Ad Format',fontsize=20)

sns.catplot(x='Ad Format',y='Clicks (all)',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Clicks on Each Ad Format',fontsize=20)

sns.catplot(x='Ad Format',y='CTR (all)',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Click Through Rate (CTR) of Each Ad Format',fontsize=20)

sns.catplot(x='Ad Format',y='Results',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Conversions from Each Ad Format',fontsize=20)

sns.catplot(x='Ad Format',y='Cost per result',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Cost per Conversion of Each Ad Format',fontsize=20)

sns.catplot(x='Ad Format',y='Result rate',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Conversion Rate of Each Ad Format',fontsize=20)


# ## 3. Effect of Targeting + Frequency + Budget on Engagement & Conversion

# In[40]:


data = data.sort_values(by=['Frequency']) 


# ### Generic View

# In[ ]:


# increase figure size 
plt.figure(figsize=(20, 6))

# subplot 1
plt.subplot(1, 6, 1)
sns.barplot(x='Frequency', y='Amount spent (INR)', data=data, estimator=np.sum,ci=None)
plt.title("Total Amount Spent")
plt.xticks(rotation=90)


# subplot 2
plt.subplot(1, 6, 2)
sns.barplot(x='Frequency', y='Clicks (all)', data=data, estimator=np.sum,ci=None)
plt.title("Total Clicks")
plt.xticks(rotation=90)

# subplot 3
plt.subplot(1, 6, 3)
sns.barplot(x='Frequency', y='CTR (all)', data=data, estimator=np.mean,ci=None)
plt.title("CTR")
plt.xticks(rotation=90)

# subplot 4
plt.subplot(1, 6, 4)
sns.barplot(x='Frequency', y='Results', data=data, estimator=np.sum,ci=None)
plt.title("Total Conversions")
plt.xticks(rotation=90)

# subplot 5
plt.subplot(1, 6, 5)
sns.barplot(x='Frequency', y='Cost per result', data=data, estimator=np.mean,ci=None)
plt.title("Avg. Cost per Conversion")
plt.xticks(rotation=90)

# subplot 6
plt.subplot(1,6, 6)
sns.barplot(x='Frequency', y='Result rate', data=data, estimator=np.mean,ci=None)
plt.title("CVR")
plt.xticks(rotation=90)


plt.tight_layout(pad=0.7)
plt.show()


# ### Granular View

# In[ ]:


sns.catplot(x='Frequency',y='Amount spent (INR)',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Amount Spent on Each Frequency',fontsize=20)

sns.catplot(x='Frequency',y='Clicks (all)',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Clicks on Each Frequency',fontsize=20)

sns.catplot(x='Frequency',y='CTR (all)',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Click Through Rate (CTR) of Each Frequency',fontsize=20)

sns.catplot(x='Frequency',y='Results',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.sum,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Total Conversions from Each Frequency',fontsize=20)

sns.catplot(x='Frequency',y='Cost per result',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Cost per Conversion of Each Frequency',fontsize=20)

sns.catplot(x='Frequency',y='Result rate',col='Ad set name',data=data,kind='bar',aspect=1, estimator=np.mean,ci=None)
plt.subplots_adjust(top=0.85)
plt.suptitle('Avg. Conversion Rate of Each Frequency',fontsize=20)


# In[ ]:




