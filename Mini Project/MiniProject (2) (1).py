#!/usr/bin/env python
# coding: utf-8

# # Customer Analysis - Retail Industry

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


Rtl_data = pd.read_csv('Online-Retail_1.csv', encoding = 'unicode_escape')
Rtl_data.head()


# In[3]:


Rtl_data.shape


# In[4]:


country_cust_data=Rtl_data[['Country','CustomerID']].drop_duplicates()
country_cust_data.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[5]:


Rtl_data = Rtl_data.query("Country=='United Kingdom'").reset_index(drop=True)


# In[6]:


Rtl_data.isnull().sum(axis=0)


# In[7]:


Rtl_data = Rtl_data[pd.notnull(Rtl_data['CustomerID'])]
Rtl_data.Quantity.min()


# In[8]:


Rtl_data.UnitPrice.min()


# In[9]:


Rtl_data = Rtl_data[(Rtl_data['Quantity']>0)]


# In[10]:


Rtl_data['InvoiceDate'] = pd.to_datetime(Rtl_data['InvoiceDate'])


# In[11]:


Rtl_data['TotalAmount'] = Rtl_data['Quantity'] * Rtl_data['UnitPrice']


# In[12]:


Rtl_data.shape


# In[13]:


Rtl_data.head()


# In[14]:


import datetime as dt
Latest_Date = dt.datetime(2011,12,10)
RFMScores = Rtl_data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (Latest_Date - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalAmount': lambda x: x.sum()})
RFMScores['InvoiceDate'] = RFMScores['InvoiceDate'].astype(int)
RFMScores.rename(columns={'InvoiceDate': 'Recency', 
                         'InvoiceNo': 'Frequency', 
                         'TotalAmount': 'Monetary'}, inplace=True)
RFMScores.reset_index().head()


# In[49]:


RFMScores.Recency.describe()


# In[16]:


import seaborn as sns
x = RFMScores['Recency']

ax = sns.distplot(x)


# In[17]:


RFMScores.Frequency.describe()


# In[18]:


import seaborn as sns
x = RFMScores.query('Frequency < 1000')['Frequency']
ax = sns.distplot(x)


# In[19]:


RFMScores.Monetary.describe()


# In[20]:


x = RFMScores.query('Monetary < 10000')['Monetary']
ax = sns.distplot(x)


# In[21]:


quantiles = RFMScores.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[22]:


quantiles


# In[23]:


def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4    
def FnMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[24]:


RFMScores['R'] = RFMScores['Recency'].apply(RScoring, args=('Recency',quantiles,))
RFMScores['F'] = RFMScores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
RFMScores['M'] = RFMScores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
RFMScores.head()


# In[25]:


RFMScores['RFMGroup'] = RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)
RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)
RFMScores.head()


# In[26]:


Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
Score_cuts = pd.qcut(RFMScores.RFMScore, q = 4, labels = Loyalty_Level)
RFMScores['RFM_Loyalty_Level'] = Score_cuts.values
RFMScores.reset_index().head()


# In[27]:


RFMScores[RFMScores['RFMGroup']=='111'].sort_values('Monetary', ascending=False).reset_index().head(10)


# In[28]:


import chart_studio as cs
import plotly.offline as po
import plotly.graph_objs as gobj
graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")
plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Frequency'],
        mode='markers',
        name='Bronze',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Silver'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Silver'")['Frequency'],
        mode='markers',
        name='Silver',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Gold'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Gold'")['Frequency'],
        mode='markers',
        name='Gold',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Frequency'],
        mode='markers',
        name='Platinum',
        marker= dict(size= 13,
            line= dict(width=1),
            color= 'black',
            opacity= 0.9
           )
    ),
]
plot_layout = gobj.Layout(
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)
graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")
plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Monetary'],
        mode='markers',
        name='Bronze',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Silver'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Silver'")['Monetary'],
        mode='markers',
        name='Silver',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Gold'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Gold'")['Monetary'],
        mode='markers',
        name='Gold',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Monetary'],
        mode='markers',
        name='Platinum',
        marker= dict(size= 13,
            line= dict(width=1),
            color= 'black',
            opacity= 0.9
           )
    ),
]
plot_layout = gobj.Layout(
        yaxis= {'title': "Monetary"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)
graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")
plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Monetary'],
        mode='markers',
        name='Bronze',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Silver'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Silver'")['Monetary'],
        mode='markers',
        name='Silver',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Gold'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Gold'")['Monetary'],
        mode='markers',
        name='Gold',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Monetary'],
        mode='markers',
        name='Platinum',
        marker= dict(size= 13,
            line= dict(width=1),
            color= 'black',
            opacity= 0.9
           )
    ),
]
plot_layout = gobj.Layout(
        yaxis= {'title': "Monetary"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)


# In[29]:


def handle_neg_n_zero(num):
    if num <= 0:
        return 1
    else:
        return num
RFMScores['Recency'] = [handle_neg_n_zero(x) for x in RFMScores.Recency]
RFMScores['Monetary'] = [handle_neg_n_zero(x) for x in RFMScores.Monetary]
Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)


# In[30]:


Recency_Plot = Log_Tfd_Data['Recency']
ax = sns.distplot(Recency_Plot)


# In[31]:


Frequency_Plot = Log_Tfd_Data.query('Frequency < 1000')['Frequency']
ax = sns.distplot(Frequency_Plot)


# In[32]:


Monetary_Plot = Log_Tfd_Data.query('Monetary < 10000')['Monetary']
ax = sns.distplot(Monetary_Plot)


# In[33]:


from sklearn.preprocessing import StandardScaler
scaleobj = StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)
Scaled_Data = pd.DataFrame(Scaled_Data, index = RFMScores.index, columns = Log_Tfd_Data.columns)


# In[34]:


from sklearn.cluster import KMeans
sum_of_sq_dist = {}
for k in range(1,15):
    km = KMeans(n_clusters= k, init= 'k-means++', max_iter= 1000)
    km = km.fit(Scaled_Data)
    sum_of_sq_dist[k] = km.inertia_
sns.pointplot(x = list(sum_of_sq_dist.keys()), y = list(sum_of_sq_dist.values()))
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Sum of Square Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[35]:


KMean_clust = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 1000)
KMean_clust.fit(Scaled_Data)
RFMScores['Cluster'] = KMean_clust.labels_
RFMScores.head()


# In[36]:


from matplotlib import pyplot as plt
plt.figure(figsize=(7,7))
Colors = ["red", "green", "blue"]
RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])
ax = RFMScores.plot(    
    kind="scatter", 
    x="Recency", y="Frequency",
    figsize=(10,8),
    c = RFMScores['Color']
)


# In[37]:


RFMScores.head(10)


# In[38]:


from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# In[39]:


Numerics=LabelEncoder()


# In[40]:


inputs=RFMScores.drop('Cluster',axis='columns')
target=RFMScores['Cluster']
target


# In[41]:


inputs['Recency_n']=Numerics.fit_transform(inputs['Recency'])
inputs['Frequency_n']=Numerics.fit_transform(inputs['Frequency'])
inputs['Monetary_n']=Numerics.fit_transform(inputs['Monetary'])
inputs['R_n']=Numerics.fit_transform(inputs['R'])
inputs['F_n']=Numerics.fit_transform(inputs['F'])
inputs['M_n']=Numerics.fit_transform(inputs['M'])
inputs['RFMGroup_n']=Numerics.fit_transform(inputs['RFMGroup'])
inputs['RFMScore_n']=Numerics.fit_transform(inputs['RFMScore'])
inputs['RFM_Loyalty_Level_n']=Numerics.fit_transform(inputs['RFM_Loyalty_Level'])
inputs['Color_n']=Numerics.fit_transform(inputs['Color'])
inputs.head(10)


# In[42]:


inputs_n=inputs.drop(['Recency','Frequency','Monetary','R','F','M','RFMGroup','RFMScore','RFM_Loyalty_Level','Color'],axis='columns')
inputs_n.head(10)


# In[43]:


Classifier=GaussianNB()
Classifier.fit(inputs_n,target)


# In[44]:


Classifier.score(inputs_n,target)


# In[45]:


Classifier.predict([[181,5,110,3,3,3,60,9,0,1]])


# In[46]:


Classifier.predict([[270,0,3862,3,3,0,57,6,3,2]])


# In[47]:


Classifier.predict([[1,102,3584,0,0,0,0,0,2,0]])


# In[48]:


print('THANK YOU')


# In[50]:


Classifier.predict([[270,0,3862,3,3,0,57,6,3,0]])

