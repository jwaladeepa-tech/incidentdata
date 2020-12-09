#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import seaborn as sns
from nltk.stem.porter import PorterStemmer

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import re  

from scipy.stats import linregress

# In[3]:


df = pd.read_excel('Incident data analysis.xlsx',sheet_name='Page 1' )


# In[4]:


df[:13879]


# In[5]:


df.info()

df.shape


# In[6]:

print("Total number of records", df.shape[0])
print("Total number of records", df.shape[1])


#number = []
#
#for i, value in enumerate(df1["Short description"]):
#    if type(value) != str:
#        number.append(i)
#number  
#
#df1.drop(df1.index[number], inplace=True)
#
#print("1 record with n/a in short description", number, "removed")  
#print("Total number of records", df1.shape[0])
#
#df1["Short description"][415]  

number = []
number = [26, 77,78,84,203, 216,248,300,324,327,415,598,705,902,928,991, 1053,1104,1107,1113,
          1152,1183,1248,1280,1330,1410,1432,1539,1562,1563,1572,1574,1603,1607, 1641,1685,1775,1819,1825,
          1827, 1908,2017,2074, 2082,2137,2150,2151,2200,2391,2403,2452,2511,2531,2556,2573,2580,
          2581, 2582,2590,2769,2822,3625,3779,3826,3846,3866,3898,3908,3961,3965,3972,3974,
          4102,4183,4312,4317,4321,4334,4346,4416,4432,4433,4438,4464,4468,4567,4585,4616,
          4756,4835,4931,5046,5297,5301,5331,5405,5432,5478,5481,5484,5599,5634,
          5847,5891,5967,5968,6156,6225,6243,6252,6254,6260, 6262,6346,6375,6416,6491,
          6495,6537,6585,6600,6610,6849,6897,6921,6928,6963,7016,7069,7098,7192,7225,7293,
          7348,7512,7542,7577,7634,7635,7771,7851,7860,7940,8104,8151,8235,8278,8302,8306,
          8308,8309,8362,8437,8439,8502,8646,8666,8676,8733,8740,8841,8842,8919,8972,
          9141,9190,9236,9240,9258,9259,9311,9324,9421,9422,9424,9426,9485,9494,9625,9638,
          9644,9689,9697,9699,9763,9772,9835,9919,9921,9966,10001,10043,10161,10162,
          10199,10217,10267,10388,10531,10532,10534,10540,10599,10612,10953,11219,11497,
          11506,11808,11965,12174,12247,12349,12351,12409,12522,12527,12649,12999,13000,13063,
          13080,13081,13341,13438,13439,13641,13749,13750,13938,14058] 

len(number)
df["Short description"][number].to_excel(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\chines.xlsx', index = False, header=True)
df.drop(df.index[number], inplace=True)
print("records with Chinese character in short description", number, "removed")  
print("Total number of records", df.shape[0])


number = []

for i, value in enumerate(df["Short description"]):
    if type(value) != str:
        number.append(i)
number  

df.drop(df.index[number], inplace=True)

print("1 record with n/a in short description", number, "removed")  
print("Total number of records", df.shape[0])

df["Short description"][number]          



# In[7]:
df = df.reset_index(drop=True)
df1 = df

df1[[ 'priority_no','priority_desc' ]] = df1.Priority.str.split("-",expand=True)


# In[8]:


df1.head()


# In[9]:


df1.info()


# In[10]:


df1['Category'].isnull().values.any()


# In[11]:


df1['Category'].isnull().sum()


# In[12]:


df1['Category'] = df1['Category'].astype('category')


# In[13]:


df1['Category_no'] = df1['Category'].cat.codes.replace(-1, np.nan)


# In[14]:


df1[320:]


# In[15]:


df1['Subcategory 1'] = df1['Subcategory 1'].astype('category')


# In[16]:


df1['Subcategory 2'] = df1['Subcategory 2'].astype('category')


# In[17]:


df1['Configuration item'] = df1['Configuration item'].astype('category')


# In[18]:


df1['Subcategory1_no'] = df1['Subcategory 1'].cat.codes.replace(-1, np.nan)


# In[19]:


df1['Subcategory2_no'] = df1['Subcategory 2'].cat.codes.replace(-1, np.nan)


# In[20]:


df1['Configuration_item_no'] = df1['Configuration item'].cat.codes.replace(-1, np.nan)


# In[21]:


df1.head()


# In[22]:


df1['updated_date'] = df['Updated'].dt.date


# In[23]:


df1['updated_time'] = df['Updated'].dt.time


# In[24]:


df1['updated_day'] = df['Updated'].dt.day


# In[25]:


df1['diff_min']  = (df1.Updated - df1.Opened)


# In[26]:


df1['diff_min']  = df1['diff_min']/np.timedelta64(1,'m')


# In[27]:


df1.head()


# In[28]:


df1['Opened_date'] = df['Opened'].dt.date


# In[29]:


df1['Opened_time'] = df['Opened'].dt.time


# In[30]:


df1['Opened_day'] = df['Opened'].dt.day


# In[31]:


df1.head()


# In[32]:


target1 = []
target2 = []
target3 = []
target4 = []
target5 = []
target6 = []
target7 = []

   
# In[34]:

target1 = []
i = 0
for idx in range(0, len(df1['Close notes'])):  
    desc = ' '
    val = df1['Close notes'][idx]
    if type(val) == str:
        temp = re.search("Issue Description: .*", val)
        aa = ()
        if temp is not None:
            tempy =  re.search('\w*\s\w*',temp.group()) 
            cc = tempy.string
            aa = cc.partition('Issue Description:')
            desc = aa[2].strip()
        else:
            desc = None
    else:
        desc = None
    target1.append(desc)
    i = i + 1
                         
df_target1 = pd.DataFrame(target1,columns = ['issue_description'])


# In[35]:
df1['issue_description'] = df_target1['issue_description']


target2 = []
i = 0
for idx in range(0, len(df1['Close notes'])):  
    desc = ' '
    val = df1['Close notes'][idx]
    if type(val) == str:
        temp = re.search("Resolved at: .*", val)
        aa = ()
        if temp is not None:
            tempy =  re.search('\w*\s\w*',temp.group()) 
            cc = tempy.string
            aa = cc.partition('Resolved at:')
            desc = aa[2].strip()
        else:
            desc = None
    else:
        desc = None
    target2.append(desc)
    i = i + 1
# In[38]:


df_target2 = pd.DataFrame(target2,columns = ['resolved_at'])


# In[39]:


df1['resolved_at'] = df_target2['resolved_at']


df1['resolved_at'] = df1['resolved_at'].str.strip()

# In[40]:


target3 = []

i = 0
for idx in range(0, len(df1['Close notes'])):  
    desc = ' '
    val = df1['Close notes'][idx]
    if type(val) == str:
        temp = re.search("Action taken: .*", val)
        aa = ()
        if temp is not None:
            tempy =  re.search('\w*\s\w*',temp.group()) 
            cc = tempy.string
            aa = cc.partition('Action taken:')
            desc = aa[2].strip()
        else:
            desc = None
    else:
        desc = None
    target3.append(desc)
    i = i + 1
# In[42]:


df_target3 = pd.DataFrame(target3,columns = ['action taken:'])


# In[43]:


df1['action taken:'] = df_target3['action taken:']

df1['action taken:'] = df1['action taken:'].str.strip()


# In[44]:

target4 = []
i = 0
for idx in range(0, len(df1['Close notes'])):  
    desc = ' '
    val = df1['Close notes'][idx]
    if type(val) == str:
        temp = re.search("Business impact: .*", val)
        aa = ()
        if temp is not None:
            tempy =  re.search('\w*\s\w*',temp.group()) 
            cc = tempy.string
            aa = cc.partition('Business impact:')
            desc = aa[2].strip()
        else:
            desc = None
    else:
        desc = None
    target4.append(desc)
    i = i + 1
# In[45]:


df_target4 = pd.DataFrame(target4,columns = ['business_impact'])


# In[46]:


df1['business_impact'] = df_target4['business_impact']

df1['business_impact'] = df1['business_impact'].str.strip()


# In[47]:

target5 = []
i = 0
for idx in range(0, len(df1['Close notes'])):  
    desc = ' '
    val = df1['Close notes'][idx]
    if type(val) == str:
        temp = re.search("Future action required: .*", val)
        aa = ()
        if temp is not None:
            tempy =  re.search('\w*\s\w*',temp.group()) 
            cc = tempy.string
            aa = cc.partition('Future action required:')
            desc = aa[2].strip()
        else:
            desc = None
    else:
        desc = None
    target5.append(desc)
    i = i + 1
# In[48]:


df_target5 = pd.DataFrame(target5,columns = ['future_action_required'])


# In[49]:


df1['future_action_required'] = df_target5['future_action_required']


df1['future_action_required'] = df1['future_action_required'].str.strip()

target6 = []
i = 0
for idx in range(0, len(df1['Close notes'])):  
    desc = ' '
    val = df1['Close notes'][idx]
    if type(val) == str:
        temp = re.search("Resolved by: .*", val)
        aa = ()
        if temp is not None:
            tempy =  re.search('\w*\s\w*',temp.group()) 
            cc = tempy.string
            aa = cc.partition('Resolved by:')
            desc = aa[2].strip()
        else:
            desc = None
    else:
        desc = None
    target6.append(desc)
    i = i + 1
# In[51]:


df_target6 = pd.DataFrame(target6,columns = ['resolved_by'])


# In[52]:


df1['resolved_by'] = df_target6['resolved_by']

df1['resolved_by'] = df1['resolved_by'].str.strip()

# In[53]:


target7 = []
i = 0
for idx in range(0, len(df1['Close notes'])):  
    desc = ' '
    val = df1['Close notes'][idx]
    if type(val) == str:
        temp = re.search("Resolution confirmed by: .*", val)
        aa = ()
        if temp is not None:
            tempy =  re.search('\w*\s\w*',temp.group()) 
            cc = tempy.string
            aa = cc.partition('Resolution confirmed by:')
            desc = aa[2].strip()
        else:
            desc = None
    else:
        desc = None
    target7.append(desc)
    i = i + 1

# In[55]:


df_target7 = pd.DataFrame(target7,columns = ['resolution_confirmed_by'])


# In[56]:


df1['resolution_confirmed_by'] = df_target7['resolution_confirmed_by']


# In[57]:


df1.head()


# In[58]:


column_names = [ 'incident_number', "short_description", "priority", "priority_desc", "priority_no", "tags", "category", 'category_no',
                 'subcategory1', 'subcategory1_no', 'subcategory2','subcategory2_no', 'configuration_item', 'configuration_item_no', 'close_notes',
                "issue_description", "action_taken", "business_impact", "resolved_at", "resolved_by", 
                "resolution_confirmed_by", "future_action_required","problem", "resolve_time", "updated", 
                "updated_date", "updated_time", "updated_day", "updates", "opened", "diff_min",
                "opened_date", "opened_time", "opened_day",  "all_categories", "all_items", "resolve_time_updates"] 


# In[59]:


column_names


# In[60]:


df_final = pd.DataFrame(columns = column_names)


# In[61]:


type(df_final)


# In[62]:


df_final['incident_number'] = df1['Number']


# In[63]:


df_final['short_description'] = df1['Short description']


# In[64]:


df_final['priority'] = df1['Priority']


# In[65]:


df_final['priority_desc'] = df1['priority_desc']
df_final['priority_no'] = df1['priority_no']
df_final['tags'] = df1['Tags']
df_final['category'] = df1['Category']
df_final['category_no'] = df1['Category_no']
df_final['subcategory1'] = df1['Subcategory 1']
df_final['subcategory1_no'] = df1['Subcategory1_no']

df_final['subcategory2'] = df1['Subcategory 2']
df_final['subcategory2_no'] = df1['Subcategory2_no']
df_final['configuration_item'] = df1['Configuration item']
df_final['configuration_item_no'] = df1['Configuration_item_no']
df_final['close_notes'] = df1['Close notes']

df_final['issue_description'] = df1['issue_description']
df_final['action_taken'] = df1['action taken:']
df_final['business_impact'] = df1['business_impact']
df_final['resolved_at'] = df1['resolved_at']
df_final['resolved_by'] = df1['resolved_by']
df_final['resolution_confirmed_by'] = df1['resolution_confirmed_by']
df_final['future_action_required'] = df1['future_action_required']

df_final['problem'] = df1['Problem']
df_final['resolve_time'] = df1['Resolve time']
df_final['updated'] = df1['Updated']
df_final['updated_date'] = df1['updated_date']
df_final['updated_time'] = df1['updated_time']
df_final['updated_day'] = df1['updated_day']
df_final['updates'] = df1['Updates']
df_final['opened'] = df1['Opened']

df_final['diff_min'] = df1['diff_min']
df_final['opened_date'] = df1['Opened_date']
df_final['opened_time'] = df1['Opened_time']

df_final['opened_day'] = df1['Opened_day']


# In[66]:


df1.columns


# In[67]:


df_final['all_categories'] =  df_final['subcategory1'].astype(str) +' '+ df_final['subcategory2'].astype(str)


# In[68]:


df_final['all_items'] =  df_final['configuration_item'].astype(str) +' '+ df_final['category'].astype(str) +' ' + df_final['subcategory1'].astype(str) +' '+ df_final['subcategory2'].astype(str)


# In[69]:


df_final.head()


# In[70]:


df_final['updated_day'] = df_final['updated'].dt.weekday_name


# In[71]:


df_final['opened_day'] = df_final['opened'].dt.weekday_name


# In[72]:


df_final.to_excel(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\17incident.xlsx', index = False, header=True)


# In[73]:


df_temp = df_final


# In[74]:


df_temp[:13878]


# In[75]:


df_temp.loc[df_temp['priority'] == 'VIP', 'priority_desc'] = 'Low'


# In[76]:


df_temp.loc[df_temp['priority'] == 'VIP', 'priority_no'] = 4


# In[77]:


df_temp.loc[df_temp['priority'] == 'VIP', 'priority'] = '4 - Low'


# In[79]:


df_temp['issue_description'][3]


# In[80]:


df_temp.loc[df_temp['short_description'] != df_temp['issue_description'], 'descriptionmatch'] = 'Not Similar'


# In[81]:


df_temp.loc[df_temp['short_description'] == df_temp['issue_description'], 'descriptionmatch'] = 'Similar'


# In[82]:


print(df_temp.pivot_table(index = ['descriptionmatch'], aggfunc='size'))


# In[89]:


df_temp.isnull().values.any()


# In[90]:


df_temp.isnull().sum().sum()


# In[91]:


df_temp.columns


# In[93]:


df_temp.loc[df_temp['short_description'].isnull()].T


# In[99]:


null_columns = df_temp.columns[df_temp.isnull().any()]


# In[100]:


null_columns


# In[101]:


df_temp[null_columns].isnull().sum()


# In[
#from nltk.tokenize import word_tokenize
#
#word_tokenize(df_temp['short_description'][24])
#
#
#df_temp['short_description_nwords'] = df_temp['short_description'].apply(lambda x: len(str(x).split(" ")))
#
#df_temp[['short_description', 'short_description_nwords']].head()

#target8 = []
#   
#for i,word in enumerate(df_temp['short_description']):
#    val = ' '
#    val = word
#    if type(word) == str:
#        a= word_tokenize(val)
#    target8.insert(i,a)

#df_target8 = pd.DataFrame(target8,columns = ['short_description_wrds'])

df_temp.columns


print('NaN values in category column')
print(df_temp['category'].isnull().sum())
print('NaN values in subcategory1 column')
print(df_temp['subcategory1'].isnull().sum())
print('NaN values in subcategory2 column')
print(df_temp['subcategory2'].isnull().sum())
print('NaN values in configuration_item column')
print(df_temp['configuration_item'].isnull().sum())


cat = df_temp['category'].isnull().sum()
subcat1 = df_temp['subcategory1'].isnull().sum()
subcat2 = df_temp['subcategory2'].isnull().sum()
config = df_temp['configuration_item'].isnull().sum()


from tabulate import tabulate

print('NaN values')
print(tabulate([['category', cat], ['subcategory1', subcat1],
                ['subcategory2', subcat2], ['configuration_item', config] ], 
               headers=['column', 'count']))


#duplicate = df_temp[df_temp.duplicated('incident_number')]
#
#duplicate
#
#
#duplicate = df_temp[df_temp.duplicated()]
#
#duplicate

#a = df_temp[df_temp.resolution_confirmed_by.apply(lambda x: x == '' )]

#df_temp['resolution_confirmed_by'] = df_temp.resolution_confirmed_by.replace(r'^\s*$', np.nan, regex=True)

df_temp.head()

#df_final.to_excel(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\11temp.xlsx', index = False, header=True)

df_temp.resolution_confirmed_by.isnull().sum()
df_temp['resolution_confirmed_by'] = df_temp.resolution_confirmed_by.replace(r'^\s*$', np.nan, regex=True)

df_temp['resolution_confirmed_by'] = df_temp['resolution_confirmed_by'].fillna(('Unassigned'))

df_temp['resolved_by'] = df_temp.resolved_by.replace(r'^\s*$', np.nan, regex=True)

df_temp.resolved_by.isnull().sum()

df_temp['resolved_by'] = df_temp['resolved_by'].fillna(('Unassigned'))


df_temp1 = df_temp

import re

#df_temp1['resolution_confirmed_by'] = df_temp1['resolution_confirmed_by'].map(lambda x: re.sub(r'\W+', '', x))

#target9 = []
#
#df_temp['problem_description'] = df_temp['short_description']
#
#for i, val in enumerate(df_temp['problem_description']):
#    if val != '':
#       if len(val) >  len(df_temp['short_description'][i]):
#        df_temp['problem_description'][i].replace(df_temp['problem_description'][i], val)   
#        
#target9[3]
#
#df_temp['problem_description'][1765]
#df_temp['issue_description'][1765]
#
#
#df_temp1['issue_description'][3]





from textblob import TextBlob



#df_temp1["sentiments"] = df_temp1["short_description"].astype(str).apply(lambda x: TextBlob(x).sentiment[0])
#
#df_temp1[["short_description", "sentiments"]].sort_values(by = "sentiments", ascending = True)
#
#df_temp1["sentiments_action"] = df_temp1["action_taken"].astype(str).apply(lambda x: TextBlob(x).sentiment[0])
#
#df_temp1[["incident_number","action_taken", "sentiments_action"]].sort_values(by = "sentiments_action", ascending = False)
#
#df_temp1["action_taken"][8109]
#
#df_temp1["sentiments_action"]
#print(df_temp1[["incident_number","action_taken", "sentiments_action"]].head(30))
#df_temp1["sentiments_action"].sort_values(ascending = False)


#df_temp1["short_nwords"] = df_temp1["short_description"].apply(lambda x: 
#    len(str(x).split(" ")))
#
#df_temp1[["incident_number","short_description","short_nwords"]].head()
#
#df_temp1["action_nwords"] = df_temp1["action_taken"].apply(lambda x: 
#    len(str(x).split(" ")))
#    
#df_temp1[["incident_number", "action_taken"]].head()   
#
#df_temp1.columns


## Importing stop words from nltk.corpus
#from nltk.corpus import stopwords
#
#stop = stopwords.words("english")
#
#df_temp1["short_description_nstopwords"] = df_temp1["short_description"].apply(lambda word: len([x for x in word.split(" ") if x in stop]))
#
#desc = df_temp1["short_description"][70]
#type(desc)
#
#
#df_test = df_temp1
#
#df_test["short_description"] = df_temp1["short_description"].apply(lambda x: x.lower())
#
#
#df_temp1["shortdesctokens"] =  df_temp1["short_description"].astype(str).apply(lambda x: TextBlob(x).words)
#
#from textblob import Word
#
#df_temp1["action_taken"].head()



#df_temp1["action_taken"] = df_temp1["action_taken"].apply(lambda x: " ".join([Word(myword).lemmatize() for myword in x.split()])  )
#aa = df_temp1["action_taken"].astype(str)
#from sklearn.feature_extraction.text import TfidfVectorizer
#vect=TfidfVectorizer(ngram_range=(2,2),lowercase=True, token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
#X = vect.fit_transform(aa)
#print(vect.get_feature_names())
#x1=X.toarray()



#corrMatrix = df_temp.corr()
#
#
## In[86]:
#
#
#print( corrMatrix )
#
#
## In[87]:
#
#
#sns.heatmap(corrMatrix, annot = True)
#plt.show
#
#
## In[88]:
#
#
#plt.matshow(corrMatrix)
#plt.show()

df_action = df_temp1

df_action['action_taken']=df_action['action_taken'].fillna(' ')
pd.set_option('display.max_colwidth',200)
df_action['action_taken'].head(10)

#import string
#
#
#def remove_punctuation(txt):
#    txt_nonpunct="".join([c for c in txt if c not in string.punctuation])
#    return txt_nonpunct
#
#df_action['action_taken'] = df_action['action_taken'].apply(lambda x: remove_punctuation(x))
#df_action['action_taken'].head()
#
#
#df_action['action_taken'] = df_action['action_taken'].str.lower()
#df_action['action_taken'].head()
#
#import nltk
#nltk.download('wordnet')
#wn=nltk.WordNetLemmatizer()
#print(wn.lemmatize('mapped'))
#print(wn.lemmatize('map'))
#
#def lemma(txt):
#    text="".join([wn.lemmatize(word) for word in txt])
#    return text
#
#df_action['action_taken'] = df_action['action_taken'].apply(lambda x : lemma(x))
#df_action['action_taken'].head()
#
#
#
#df_action["action_taken"] = df_action["action_taken"].apply(lambda x: " ".join([Word(myword).lemmatize() for myword in x.split()])  )
#df_action['action_taken'].head()
#
#actionstr = df_action['action_taken'].astype(str)
#actionstr
#
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#vect=TfidfVectorizer(ngram_range=(3,3),lowercase=False,stop_words='english')
#X = vect.fit_transform(actionstr)
#vect.get_feature_names()
#
#frequencies = sum(X).toarray()[0]
#tri_grams_df = pd.DataFrame(frequencies, index=vect.get_feature_names(), columns=['action_trifrequency'])
#tri_grams_df
#
#tri_grams_df['action_trifrequency'][1:]
#
#tri_grams_df.sort_values(by = "action_trifrequency",ascending=False).head(20)
#
#tri_grams_df.to_csv(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\action_tri_grams.csv', index = True, header=True)
#
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#vect=TfidfVectorizer(ngram_range=(1,1),lowercase=False,stop_words='english')
#X = vect.fit_transform(actionstr)
#vect.get_feature_names()
#
#frequencies = sum(X).toarray()[0]
#one_grams_df = pd.DataFrame(frequencies, index=vect.get_feature_names(), columns=['action_onefrequency'])
#one_grams_df
#
#one_grams_df['action_onefrequency'][1:]
#
#one_grams_df.sort_values(by = "action_onefrequency",ascending=False).head(20)
#
#one_grams_df.to_csv(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\action_one_grams.csv', index = True, header=True)
#
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#vect=TfidfVectorizer(ngram_range=(3,3),lowercase=False,stop_words='english')
#X = vect.fit_transform(actionstr)
#vect.get_feature_names()
#
#frequencies = sum(X).toarray()[0]
#bi_grams_df = pd.DataFrame(frequencies, index=vect.get_feature_names(), columns=['action_bifrequency'])
#bi_grams_df
#
#bi_grams_df['action_bifrequency'][1:]
#
#bi_grams_df.sort_values(by = "action_bifrequency",ascending=False).head(20)
#
#bi_grams_df.to_csv(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\action_bi_grams.csv', index = True, header=True)
#


df_action['problem_description'] = df_action['short_description']

df_action['problem_description']

for i, row in df_action.iterrows():
    if(pd.isnull(row['short_description'])):
        if(pd.isnull(row['issue_description'])):
            print('Drop: ', row['incident_number'])
            print('----------------------------------------')
#           df_test = df_test.drop[row['incident_number']]
#         else:
#             print('Missing SD: ', row['short_description'])
#             print('===========================')
    elif(pd.isnull(row['issue_description'])):
        print('Append: ', row['short_description'])
        df_action['problem_description'][i] = row['short_description']
    elif len(str(row['short_description'])) > len(str(row['issue_description'])):
        print('Short description is longer')
        df_action['problem_description'][i] = row['short_description']
    elif len(str(row['issue_description'])) >= len(str(row['short_description'])):
        print('Issue description is longer than or equal to the short description.')
        df_action['problem_description'][i] = row['issue_description']
        
        
df_action.to_excel(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\23temp.xlsx', index = False, header=True)        
        
df_action['problem_description'].head()     

#actionstr = df_action['problem_description'].astype(str)
#actionstr
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#vect=TfidfVectorizer(ngram_range=(3,3),lowercase=False,stop_words='english')
#X = vect.fit_transform(actionstr)
#vect.get_feature_names()
#
#frequencies = sum(X).toarray()[0]
#tri_grams_df = pd.DataFrame(frequencies, index=vect.get_feature_names(), columns=['problem_trifrequency'])
#tri_grams_df
#
#tri_grams_df['problem_trifrequency'][1:]
#
#tri_grams_df.sort_values(by = "problem_trifrequency",ascending=False).head(20)
#
#tri_grams_df.to_csv(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\problem_tri_grams.csv', index = True, header=True)     
#tri_grams_df
#
#
#vect=TfidfVectorizer(ngram_range=(2,2),lowercase=False,stop_words='english')
#X = vect.fit_transform(actionstr)
#vect.get_feature_names()
#
#frequencies = sum(X).toarray()[0]
#bi_grams_df = pd.DataFrame(frequencies, index=vect.get_feature_names(), columns=['problem_bifrequency'])
#bi_grams_df
#
#bi_grams_df['problem_bifrequency'][1:]
#
#bi_grams_df.sort_values(by = "problem_bifrequency",ascending=False).head(20)
#
#bi_grams_df.to_csv(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\problem_bi_grams.csv', index = True, header=True)     
#bi_grams_df
#
#
#vect=TfidfVectorizer(ngram_range=(2,2),lowercase=False,stop_words='english')
#X = vect.fit_transform(actionstr)
#vect.get_feature_names()
#print(X.toarray())
#
#frequencies = sum(X).toarray()[0]
#one_grams_df = pd.DataFrame(frequencies, index=vect.get_feature_names(), columns=['problem_onefrequency'])
#one_grams_df
#
#one_grams_df['problem_onefrequency'][1:]
#
#one_grams_df.sort_values(by = "problem_onefrequency",ascending=False).head(20)
#
#one_grams_df.to_csv(r'C:\Users\jw20093543\Documents\focused run\HANA APL OBJECTS\problem_one_grams.csv', index = True, header=True)     
#one_grams_df
#

len(df_action)

df_action.columns


# Sentiments for TextBlob

df_action["sentiments"] = df_action['action_taken'].apply(lambda x: TextBlob(x).sentiment[0])

df_action[["action_taken","sentiments"]].sort_values(by = "sentiments",ascending = True)




sentiment_scores_tb = [round(TextBlob(doc).sentiment.polarity,3) for doc in df_action["action_taken"]]
sentiment_scores_tb

sentiment_category_tb = ['positive' if score > 0
                         else 'negative' if score < 0
                         else 'neutral' for score in sentiment_scores_tb]


sentiment_category_tb

result_tb = pd.DataFrame([list(df_action['incident_number']),sentiment_scores_tb, sentiment_category_tb]).T

result_tb
result_tb.columns = ['incident_no', 'sentiment_scores_tb', 'sentiment_category_tb']

result_tb['sentiment_scores_tb'] = result_tb.sentiment_scores_tb.astype('float')

result_tb.groupby(by=['sentiment_category_tb']).describe()

result_tb.head()

# Sentiment for Afill 

from afinn import Afinn

corpus = list(df_action['action_taken'])

af = Afinn()

sentiment_score_afinn = [af.score(doc) for doc in corpus]
sentiment_score_afinn

sentiment_category_afinn = ['positive' if score > 0
                         else 'negative' if score < 0
                         else 'neutral' for score in sentiment_score_afinn]


sentiment_category_afinn


result_afinn = pd.DataFrame([list(df_action['incident_number']),sentiment_score_afinn, sentiment_category_afinn]).T

result_afinn
result_afinn.columns = ['incident_no', 'sentiment_score_afinn', 'sentiment_category_afinn']

result_afinn['sentiment_score_afinn'] = result_afinn.sentiment_score_afinn.astype('float')

result_afinn.groupby(by=['sentiment_category_afinn']).describe()

result_afinn.head()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer =  SentimentIntensityAnalyzer()

sentiment_scores_vad = [analyzer.polarity_scores(doc) for doc in corpus]

result_vad = pd.DataFrame(sentiment_scores_vad)

result_vad.head()

compound_score = list(result_vad['compound'])

compound_score

sentiment_category_vad = ['positive' if score > 0.050
                          else 'negative' if score < -0.05
                          else 'neutral'
                          for score in compound_score]

result_vad_new = pd.DataFrame([list(df_action['incident_number']),compound_score, sentiment_category_vad]).T

result_vad_new.columns = ['incident_no', 'compound_score', 'sentiment_category_vad']

result_vad_new['compound_score'] = result_vad_new.compound_score.astype('float')

result_vad_new.groupby(by=['sentiment_category_vad']).describe()

result_afinn.groupby(by=['sentiment_category_afinn']).describe()

result_tb.groupby(by=['sentiment_category_tb']).describe()

Final_df = result_tb
Final_df['sentiment_score_afinn']  = result_afinn['sentiment_score_afinn']
Final_df['sentiment_category_afinn']  = result_afinn['sentiment_category_afinn']

Final_df['sentiment_score_vad'] = result_vad_new['compound_score']
Final_df['sentiment_category_vad'] = result_vad_new['sentiment_category_vad']


Final_df.head(20)

tar = []


for i, rec in Final_df.iterrows():
    a = ' '
    
    if ( Final_df['sentiment_category_tb'][i] == 'positive' 
        and Final_df['sentiment_category_afinn'][i] == 'positive' 
        and Final_df['sentiment_category_vad'][i] == 'positive' ):
         a = 'positive'
    elif ( Final_df['sentiment_category_tb'][i] == 'negative'
     and Final_df['sentiment_category_afinn'][i] == 'negative'
     and Final_df['sentiment_category_vad'][i] == 'negative' ):
         a = 'negative'
    else:
         a = 'neutral'
    tar.append(a)
    
tar    
    
df_tar = pd.DataFrame(tar,columns = ['final_sentiment'])

tar = []


Final_df['final_sentiment'] = df_tar['final_sentiment'] 
Final_df['action_taken']  = df_action['action_taken']
Final_df['resolve_time'] = df_action['resolve_time']
Final_df['updates'] = df_action['updates']
Final_df['all_items'] = df_action['all_items']
Final_df['incident_no'] = df_action['incident_number']

Final_df.info()
import datetime
#Final_df['resolve_days'] = Final_df['resolve_time'].apply(lambda x: datetime.timedelta(seconds = x))
#Final_df['resolve_days']


Final_df.to_excel('sentiment.xlsx',index = True, header=True)


print(df_action['all_items'].unique())


df_mean = Final_df.groupby(['all_items']).agg({'incident_no': "count",
                         'resolve_time':[pd.np.min, pd.np.max, pd.np.mean,pd.np.std],
                         'updates':[pd.np.min, pd.np.max, pd.np.mean, pd.np.std]})
    
df_mean.to_excel('df_mean.xlsx')



Final_df.columns

df_mean.columns.values

a = df_mean.index.values

a
mean = pd.DataFrame(a)

mean.columns

mean[0].head()

#for i,col in enumerate(col):

mean['all_items'] = mean[0]
mean.columns

mean.drop(0,inplace=True, axis=1)
mean.head()

df_mean['resolve_time']['mean'][0]

ret = []
for i,col in enumerate(df_mean['resolve_time']['mean']):
    z = df_mean['resolve_time']['mean'][i]
    ret.append(z)

    
mean['Mean of resolve_time'] = pd.DataFrame(ret)          

mean.head()

ret = []
for i,col in enumerate(df_mean['updates']['mean']):
    z = df_mean['updates']['mean'][i]
    ret.append(z)

    
mean['Mean of updates'] = pd.DataFrame(ret) 

mean.head() 


ret = []
for i,col in enumerate(df_mean['resolve_time']['std']):
    z = df_mean['resolve_time']['std'][i]
    ret.append(z)

    
mean['Std of resolve_time'] = pd.DataFrame(ret)          

mean.head()

ret = []
for i,col in enumerate(df_mean['updates']['std']):
    z = df_mean['updates']['std'][i]
    ret.append(z)

    
mean['Std of updates'] = pd.DataFrame(ret) 
        

mean['sqrt2cols'] = np.sqrt(mean['Mean of resolve_time'] * mean['Mean of updates'])
mean['sqrt2cols'].head()

mean['all_items']

Final_df['dispresolve_mean'] = Final_df['resolve_time']
Final_df['dispupdates_mean'] = Final_df['updates']
Final_df['mean_resolve'] =  Final_df['resolve_time']
Final_df['mean_updates'] =  Final_df['updates']
Final_df['cv_resolve_time'] = Final_df['resolve_time']
Final_df['cv_updates'] = Final_df['updates']
Final_df['cv_sqrt2cols'] = Final_df['updates']

mean['cv_resolve_time'] = mean['Std of resolve_time'] / mean['Mean of resolve_time']
mean['cv_updates'] = mean['Std of updates'] / mean['Mean of updates']
mean['cv_sqrt2cols'] = np.sqrt(mean['cv_resolve_time'] * mean['cv_updates'])

mean.columns
Final_df.columns


disp_resolve = []
disp_updates = []
for i, val in enumerate(Final_df['all_items']):
    temp = mean[mean['all_items'] == Final_df['all_items'][i]]
    Final_df.at[i,'dispresolve_mean'] = Final_df['resolve_time'][i]-temp['Mean of resolve_time']
    Final_df.at[i,'dispupdates_mean'] = Final_df['updates'][i]-temp['Mean of updates']
    Final_df.at[i,'mean_resolve'] = temp['Mean of resolve_time']
    Final_df.at[i,'mean_updates'] = temp['Mean of updates']
  
    
for i, val in enumerate(Final_df['all_items']):
    temp = mean[mean['all_items'] == Final_df['all_items'][i]]
#    Final_df.at[i,'dispresolve_mean'] = Final_df['resolve_time'][i]-temp['Mean of resolve_time']
    #Final_df.at[i,'dispupdates_mean'] = Final_df['updates'][i]-temp['Mean of updates']
    Final_df.at[i,'mean_resolve'] = temp['Mean of resolve_time']
    Final_df.at[i,'mean_updates'] = temp['Mean of updates']   
    

mean.to_excel('mean27.xlsx')

mean['cv_updates']
for i, val in enumerate(Final_df['all_items']):
    temp = mean[mean['all_items'] == Final_df['all_items'][i]]
    Final_df.at[i,'cv_resolve_time'] = temp['cv_resolve_time']
   

   
Final_df.info()

Final_df.to_excel('Final_Datamean_dispersion.xlsx')


df_action.columns

df_action.shape
Final_df

df_action.to_excel('dataset_withoutmean.xlsx')

df_action['mean_resolve'] = Final_df['mean_resolve']
df_action['mean_updates'] = Final_df['mean_updates']
df_action['dispresolve_mean'] = Final_df['dispresolve_mean']
df_action['dispupdates_mean'] = Final_df['dispupdates_mean']
df_action['final_sentiment'] = Final_df['final_sentiment']



Final_df.columns

df_action['sentiment_scores_textblob'] = result_tb['sentiment_scores_tb']
df_action['sentiment_category_textblob'] = Final_df['sentiment_category_tb']
df_action['sentiment_scores_afinn'] = result_afinn['sentiment_score_afinn']
df_action['sentiment_category_afinn'] = Final_df['sentiment_category_afinn']
df_action['sentiment_scores_vader'] = result_vad_new['compound_score']
df_action['sentiment_category_vader'] = Final_df['sentiment_category_vad']

df_action.columns

df_action = df_action.drop(['descriptionmatch','sentiments'], axis = 1)

df_action.to_excel('dataset_withmean_scores.xlsx')



df_temp = Final_df
Final_df = Final_df.drop(['sentiment_scores_tb','sentiment_score_afinn','sentiment_score_vad'], axis =1)
Final_df.to_excel('Final_df.xlsx')

corr_mean = Final_df.corr()

corr_mean   

sns.heatmap(corr_mean) 


sns.pairplot(Final_df)

  
result = Final_df.groupby(['all_items','final_sentiment']).agg({'incident_no': "count",
                         'resolve_time':[pd.np.min, pd.np.max, pd.np.mean,pd.np.std],
                         'updates':[pd.np.min, pd.np.max, pd.np.mean, pd.np.std]})

print(result)

result.to_excel('temp.xlsx')

result.columns.values[0][1]

b = result.columns.values

b[0]


columns = ['count of incidentno', 'Average of Updates',
           'Min of Updates','Max of Updates','Std of Updates',
           'Average of resolve_time','Min of resolve_time','Max of resolve_time',
           'Std of resolve_time']


a = result.index.values

df_result = pd.DataFrame(a)

#for i,col in enumerate(col):

df_result['all_items'] = df_result[0].apply(lambda x: x[0])
df_result['all_items']

df_result['sentiment_ctag'] = df_result[0].apply(lambda x: x[1])
    
df_result['sentiment_ctag']


df_result.drop(0,inplace=True, axis=1)
df_result.head()

result.columns

ret = []
for i,col in enumerate(result['incident_no']['count']):
    z = result['incident_no']['count'][i]
    ret.append(z)
    
df_result['count of incidentno'] = pd.DataFrame(ret)  

df_result['count of incidentno'].head()

 
ret = []
for i,col in enumerate(result['resolve_time']['amin']):
    z = result['resolve_time']['amin'][i]
    ret.append(z)

    
df_result['Min of resolve_time'] = pd.DataFrame(ret)  

ret = []
for i,col in enumerate(result['resolve_time']['amax']):
    z = result['resolve_time']['amax'][i]
    ret.append(z)

    
df_result['Max of resolve_time'] = pd.DataFrame(ret)          

df_result.head()

ret = []
for i,col in enumerate(result['resolve_time']['mean']):
    z = result['resolve_time']['mean'][i]
    ret.append(z)

    
df_result['Average of resolve_time'] = pd.DataFrame(ret)          

df_result.head()


ret = []
for i,col in enumerate(result['resolve_time']['std']):
    z = result['resolve_time']['std'][i]
    ret.append(z)

    
df_result['Std of resolve_time'] = pd.DataFrame(ret)          

df_result.head()


# In[56]:
ret = []
for i,col in enumerate(result['updates']['amin']):
    z = result['updates']['amin'][i]
    ret.append(z)

    
df_result['Min of updates'] = pd.DataFrame(ret) 

df_result.head() 



ret = []
for i,col in enumerate(result['updates']['amax']):
    z = result['updates']['amax'][i]
    ret.append(z)

    
df_result['Max of updates'] = pd.DataFrame(ret)          

df_result.head()

ret = []
for i,col in enumerate(result['updates']['mean']):
    z = result['updates']['mean'][i]
    ret.append(z)

    
df_result['Average of updates'] = pd.DataFrame(ret)          

df_result.head()


ret = []
for i,col in enumerate(result['updates']['std']):
    z = result['updates']['std'][i]
    ret.append(z)

    
df_result['Std of updates'] = pd.DataFrame(ret)          

df_result.head()

df_result.to_excel('temp1.xlsx')

corr = df_result.corr()

sns.heatmap(corr)
corr.to_excel('corr.xlsx')
sns.scatterplot(x= df_result['all_items'], y = df_result['count of incidentno'])


Final_df.columns

Final_df.to_excel('polarity_scores.xlsx')

df_result.columns

df_result['sqrt2cols'] = np.sqrt(df_result['Average of resolve_time'] * df_result['Average of updates'])
df_result['sqrt2cols'].head()

df_positive = df_result[df_result['sentiment_ctag'] == 'positive' ]
df_positive= df_positive.drop(['count of incidentno',
       'Min of resolve_time', 'Max of resolve_time','Std of resolve_time', 'Min of updates', 'Max of updates',
       'Std of updates'], axis = 1)
    
df_positive.columns    

corr_positive = df_positive.corr()
print(corr_positive)
sns.heatmap(corr_positive)


df_negative = df_result[df_result['sentiment_ctag'] == 'negative']
df_negative = df_negative.drop(['count of incidentno',
       'Min of resolve_time', 'Max of resolve_time','Std of resolve_time', 'Min of updates', 'Max of updates',
       'Std of updates'], axis = 1)
    
df_negative.columns    

corr_negative = df_negative.corr()
print(corr_negative)

sns.heatmap(corr_negative)

df_neutral = df_result[df_result['sentiment_ctag'] == 'neutral']
df_neutral = df_neutral.drop(['count of incidentno',
       'Min of resolve_time', 'Max of resolve_time','Std of resolve_time', 'Min of updates', 'Max of updates',
       'Std of updates'], axis = 1)
    
df_neutral.columns    

corr_neutral = df_neutral.corr()
print(corr_neutral)

sns.heatmap(corr_neutral)

print('Positive polarity correlation scores')
print(corr_positive)
print(sns.heatmap(corr_positive))

print('Negative polarity correlation scores')
print(corr_negative)

sns.heatmap(corr_negative)

print('Neutral polarity correlation scores')

sns.heatmap(corr_neutral)

corr_positive.to_excel('corr_positive.xlsx')
corr_negative.to_excel('corr_negative.xlsx')
corr_neutral.to_excel('corr_neutral.xlsx')

#df_action.columns
#Final_df.columns
#
#Final_df[]
#
#result = Final_df.groupby(['all_items','final_sentiment']).agg({'incident_no': "count",
#                         'resolve_time':[pd.np.min, pd.np.max, pd.np.mean,pd.np.std],
#                         'updates':[pd.np.min, pd.np.max, pd.np.mean, pd.np.std]})

df_result.columns
df_result['Std of resolve_time'].head()
df_result['cv_resolve_time'] = df_result['Std of resolve_time'] / df_result['Average of resolve_time']
df_result['cv_updates'] = df_result['Std of updates'] / df_result['Average of updates']
df_result['cv_sqrt2cols'] = np.sqrt(df_result['cv_resolve_time'] * df_result['cv_updates'])

df_result['cv_resolve_time'].head()
df_result['cv_updates'].head()
df_result['cv_sqrt2cols']


df_cv = df_result
df_cv= df_cv.drop(['count of incidentno',
       'Min of resolve_time', 'Max of resolve_time','Std of resolve_time', 'Min of updates', 'Max of updates',
       'Std of updates'], axis = 1)

df_cv.head()    
corr_cv = df_cv.corr()
print(corr_cv)
sns.heatmap(corr_cv)

df_cv.to_excel('cv_corr.xlsx')
corr_cv.to_excel('cv_corr.xlsx', sheet_name = 'Sheet2')

#df_action.columns
Final_df.columns

sns.pairplot(df_cv)

pos = df_cv[df_cv['sentiment_ctag'] == 'positive']
sns.pairplot(pos)

neg = df_cv[df_cv['sentiment_ctag'] == 'negative']
sns.pairplot(neg)


neu = df_cv[df_cv['sentiment_ctag'] == 'neutral']
sns.pairplot(neu)
#
#result = Final_df.groupby(['all_items','final_sentiment']).agg({'incident_no': "count",
#                         'resolve_time':[pd.np.min, pd.np.max, pd.np.mean,pd.np.std],
#                         'updates':[pd.np.min, pd.np.max, pd.np.mean, pd.np.std]})


Final_df.columns



df_action = df_action.drop(['short_description_nwords','short_nwords', 'action_nwords',
                'short_description_nstopwords', 'sentiments'], axis = 1)
    

df_action.columns
df_action.info()

df_action = df_action.drop(['shortdesctokens', 'resolve_time_updates','problem'],axis = 1)

df_action['all_items'].head()


df_action['sentiment_tag'] = Final_df['final_sentiment']

df_action['category_no'] = df_action['category_no'].astype('category')
df_action['priority_no'] = df_action['priority_no'].astype('category')
df_action['sentiment_tag'] = df_action['sentiment_tag'].astype(str)
df_action['res_hours']  = df_action['resolve_time']/3600

type(df_action['res_hours'][0])
df_action['res_hours']  = df_action['res_hours']/np.timedelta64(1,'h')
df_action = df_action.astype('object')
sns.pairplot(df_action, hue="sentiment_tag")
#plt.show()   
df_action.to_excel('df_action.xlsx')

df_action['category'] = df_action['category'].astype('object')











  
    
    








































        




















































 
 

















    

    
   

