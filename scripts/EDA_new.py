#!/usr/bin/env python
# coding: utf-8

# In[61]:


# Warning: please change the path of csv files before you run the code
# Warning: please make sure your version of Python and Python package are satisfied with requirement.


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import datetime
print(datetime.datetime.now())


# ## Read data from local (please change the path of csvs)

# In[3]:


property_2016 = pd.read_csv('../data/properties_2016.csv')
property_2016.head()


# In[4]:


property_2017 = pd.read_csv('../data/properties_2017.csv')
property_2017.head()


# ## Prepare raw dataset from Kaggle data

# In[5]:


df = pd.concat([property_2016, property_2017], axis = 0)
df.shape


# ### drop two columns which have linear relationship with y value (prediction)

# In[6]:


df.drop(['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt'], axis = 1, inplace = True)
df.shape


# In[7]:


df.columns


# In[8]:


df['assessmentyear'].value_counts()


# In[9]:


# only use the latest data
df = df[df['assessmentyear'] == 2016]
df.drop('assessmentyear', axis = 1, inplace = True)


# In[10]:


df.shape


# In[11]:


# y value: taxvaluedollarcn
# remove nan of y value
df = df[df['taxvaluedollarcnt'].notnull()]
df.shape


# In[12]:


df = df[df['taxvaluedollarcnt'] > 10000]
df.shape


# In[13]:


df['y'] = df['taxvaluedollarcnt']
df.drop('taxvaluedollarcnt', axis = 1, inplace = True)
df.shape


# In[14]:


df.columns


# # EDA

# In[15]:


# remove
df['propertyzoningdesc'].value_counts()


# In[16]:


# remove
df['rawcensustractandblock'].value_counts()


# In[17]:


# remove
df['censustractandblock'].value_counts()


# In[18]:


# remove
df['regionidneighborhood'].value_counts()


# In[19]:


print('before drop: ', df.shape)
df.drop(['parcelid', 'propertycountylandusecode', 'propertyzoningdesc', 'rawcensustractandblock',
         'censustractandblock', 'regionidneighborhood'], axis = 1, inplace = True)
print('after drop: ', df.shape)


# In[20]:


for col in df.columns:
    print(col)
    print(df[col].value_counts())
    print("==================================")


# In[21]:


cat_col = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid',
           'decktypeid', 'fips', 'hashottuborspa', 'heatingorsystemtypeid', 'poolcnt', 'poolsizesum', 
           'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid', 'regionidcounty', 
           'storytypeid', 'typeconstructiontypeid', 'fireplaceflag', 'taxdelinquencyflag']

numeric_col = ['basementsqft', 'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'finishedfloor1squarefeet', 
               'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13', 
               'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt', 
               'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'latitude', 'longitude', 'lotsizesquarefeet', 
               'roomcnt', 'threequarterbathnbr', 'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 
               'yearbuilt', 'numberofstories', 'regionidcity', 'taxamount', 'taxdelinquencyyear', 'regionidzip']


# In[24]:


print('length of category column: ', len(cat_col))


# In[23]:


print('length of numeric column: ', len(numeric_col))


# In[25]:


print(df.info())


# In[26]:


df.reset_index(inplace = True, drop = True)


# ## Data cleaning: Numeric columns   
# (1) drop columns which have more than 90% of missing values.   
# (2) fill nan values with mean values of each columns.

# In[27]:


cat_property = df[cat_col]
numeric_property = df[numeric_col]


# In[28]:


total = numeric_property.isnull().sum().sort_values(ascending=False)
percent = (numeric_property.isnull().sum()/numeric_property.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(31)


# In[29]:


# remove the columns the percentage of null value of which is larger than 90%
df.drop(['basementsqft', 'yardbuildingsqft26', 'finishedsquarefeet13', 'finishedsquarefeet6',
        'taxdelinquencyyear', 'yardbuildingsqft17', 'finishedsquarefeet15', 'finishedfloor1squarefeet',
        'finishedsquarefeet50'], axis = 1, inplace = True)


# In[30]:


numeric_col = []
for col in df.columns:
    if col not in cat_col:
        numeric_col.append(col)
print(numeric_col)


# In[31]:


numeric_property = df[numeric_col]
total = numeric_property.isnull().sum().sort_values(ascending=False)
percent = (numeric_property.isnull().sum()/numeric_property.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# In[32]:


# replace null with mean value
df[numeric_col] = df[numeric_col].fillna(df[numeric_col].mean())


# In[33]:


numeric_property = df[numeric_col]
total = numeric_property.isnull().sum().sort_values(ascending=False)
percent = (numeric_property.isnull().sum()/numeric_property.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# ## Category columns

# In[34]:


cat_property = df[cat_col]
total = cat_property.isnull().sum().sort_values(ascending=False)
percent = (cat_property.isnull().sum()/cat_property.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(len(cat_col))


# BuildingClassTypeID  
# 1	Buildings having fireproofed structural steel frames carrying all wall, floor and roof loads. Wall, floor and roof structures are built of non-combustible materials.  
# 2	Buildings having fireproofed reinforced concrete frames carrying all wall floor and roof loads which are all non-combustible.  
# 3	Buildings having exterior walls built of a non-combustible material such as brick, concrete, block or poured concrete. Interior partitions and roof structures are built of combustible materials. Floor may be concrete or wood frame.  
# 4	Buildings having wood or wood and steel frames  
# 5	Specialized buildings that do not fit in any of the above categories!  

# In[37]:


# not an important feature - remove
df.drop(['buildingclasstypeid'], axis = 1, inplace = True)


# In[38]:


cat_col.remove('buildingclasstypeid')
cat_col


# In[39]:


# 7: Basement
df['storytypeid'].value_counts()
df.drop(['storytypeid'], axis = 1, inplace = True)
cat_col.remove('storytypeid')


# In[40]:


# replace binary column with False/0
# remove other category columns

tmp_col = ['architecturalstyletypeid', 'typeconstructiontypeid', 'decktypeid', 'pooltypeid10', 'poolsizesum', 
          'pooltypeid2', 'hashottuborspa', 'taxdelinquencyflag']
for i in tmp_col:
    print(df[i].value_counts())


# In[41]:


df.drop(['architecturalstyletypeid', 'typeconstructiontypeid', 'decktypeid', 'poolsizesum'], axis = 1, inplace = True)
df.shape


# In[42]:


cat_col = []
for item in df.columns:
    if item not in numeric_col:
        cat_col.append(item)


# In[43]:


for i in range(0, len(cat_col)):
    column_name = cat_col[i]
    print(df[column_name].value_counts())


# In[44]:


# 5 stands for None
df['airconditioningtypeid'].fillna(5, inplace = True)
df['airconditioningtypeid'].value_counts()


# In[45]:


# buildingqualitytypeid: Overall assessment of condition of the building from best (lowest) to worst (highest)
df['buildingqualitytypeid'].fillna(df['buildingqualitytypeid'].mode()[0], inplace = True)
df['buildingqualitytypeid'].value_counts()


# In[46]:


df['fips'].fillna(df['fips'].mode()[0], inplace = True)
df['fips'].value_counts()


# In[47]:


df['hashottuborspa'].fillna(False, inplace = True)
df['hashottuborspa'].value_counts()


# In[48]:


# 13 stands for None
df['heatingorsystemtypeid'].fillna(13, inplace = True)
df['heatingorsystemtypeid'].value_counts()


# In[49]:


for item in ['poolcnt', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'fireplaceflag', 'taxdelinquencyflag']:
    df[item].fillna(0, inplace = True)


# In[50]:


# 261: SFR
df['propertylandusetypeid'].fillna(261, inplace = True)
df['propertylandusetypeid'].value_counts()


# In[51]:


df['regionidcity'].fillna(df['regionidcity'].mode()[0], inplace = True)
df['regionidcounty'].fillna(df['regionidcounty'].mode()[0], inplace = True)
df['regionidzip'].fillna(df['regionidzip'].mode()[0], inplace = True)


# In[52]:


cat_property = df[cat_col]
total = cat_property.isnull().sum().sort_values(ascending=False)
percent = (cat_property.isnull().sum()/cat_property.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# In[53]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# ## Data analysis

# In[55]:


# y value distribution
ulimit = np.percentile(df.y.values, 99)
llimit = np.percentile(df.y.values, 1)
df['y'].iloc[df['y']>ulimit] = ulimit
df['y'].iloc[df['y']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(df.y.values, bins=50, kde=False)
plt.xlabel('y', fontsize=12)
plt.show()


# In[56]:


# longtitude and latitude
# From the data page, we are provided with a full list of real estate properties in three counties 
# (Los Angeles, Orange and Ventura, California) data in 2016.
plt.figure(figsize=(12,12))
sns.jointplot(x=df.latitude.values, y=df.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


# In[57]:


#correlation matrix for numeric df
numeric_property = df[numeric_col]
corrmat = numeric_property.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1.0, square=True)


# In[58]:


#correlation matrix: feature vs logerror
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap = "Accent")


# According to the crystal ball, these are ten variables most correlated with 'logerror'.  
# Tax amount and finishedsqaurefeet12 are two of the most columns which are closest related to house price.       
# It is reasonable because the house price will be larger if the tax amount is higher or finished sqft of the houses is  larger.   

# In[59]:


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'y')['y'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:




