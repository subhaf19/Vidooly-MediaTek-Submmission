
# coding: utf-8

# In[197]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score


# In[219]:


df = pd.read_csv(r"C:\Users\Piyush\Desktop\machine_learning\ad_org\data\mn\ad_org_train.csv")


# In[220]:


df


# In[221]:


df.dtypes


# In[222]:


df = df.drop(columns = ["vidid"])


# In[224]:


## Splitting the date into year, month and day
p_year = []
p_month = []
p_day = []


# In[225]:


for i in range(0, len(df)):
    p_year.append(df['published'][i][0:4])
    p_month.append(df['published'][i][5:7])
    p_day.append(df['published'][i][9:])


# In[226]:


df['p_year'] = p_year
df['p_month'] = p_month
df['p_day'] = p_day


# In[227]:


df['p_year'] = df['p_year'].astype(str).astype(int)


# In[228]:


df['p_month'] = df['p_month'].astype(str).astype(int)
df['p_day'] = df['p_day'].astype(str).astype(int)


# In[229]:


#Splitting the duration into hour, min and seconds
hr =[]
mins = []
sec = []


# In[230]:


for i in range(0, len(df)):
    
    temp1 = df['duration'][i].find('H')
    if(temp1==-1):
        hr.append('0')
        flag = df['duration'][i].find('M')
        if(flag!=-1):
            temp2 = df['duration'][i].partition('M')[0]
            mins.append(temp2[2:])
            temp3 = df['duration'][i].partition('M')[2]
            sec.append(temp3[:-1])
        else:
            mins.append('0')
            temp2 = df['duration'][i].partition('S')[0]
            sec.append(temp2[2:])
    else:
        temp2 = df['duration'][i].partition('H')[0]
        hr.append(temp2[2:])
        temp3 = df['duration'][i].partition('H')[2]
        flag = temp3.find('M')
        if(flag != -1):
            temp4 = temp3.partition('M')[0]
            mins.append(temp4)
            temp5 = temp3.partition('M')[2]
            sec.append(temp5[:-1])
        else:
            mins.append('0')
            temp5 = temp3.partition('S')[0]
            sec.append(temp5)


# In[233]:


df['hr'] = hr
df['mins'] = mins
df['sec'] = sec


# In[234]:


#Filling NA values with 0
def convert_fill(df):
    return df.stack().apply(pd.to_numeric, errors='ignore').fillna(0).unstack()

df = convert_fill(df)


# In[237]:


df['hr'] = df['hr'].astype(str).astype(int)
df['mins'] = df['mins'].astype(str).astype(int)
df['sec'] = df['sec'].astype(str).astype(int)


# In[242]:


df = df.drop(columns = ["published", "duration"])


# In[243]:


df.dtypes


# In[231]:


#Mapping F value in different field to -1
mapping = {'F': -1}
df = df.replace({'likes': mapping, 'dislikes': mapping, 'comment': mapping, 'views':mapping})


# In[213]:


df.index[df['views'] == 'F'].tolist()
df['views'][7447] = -1
df['views'][8112] = -1


# In[238]:


#Label Encoding of Category variable
le = preprocessing.LabelEncoder()
le.fit(df['category'])
df['category'] = le.transform(df['category'])


# In[239]:


df['adview'] = df['adview'].astype(str).astype(int)
df['views'] = df['views'].astype(str).astype(int)
df['likes'] = df['likes'].astype(str).astype(int)
df['dislikes'] = df['dislikes'].astype(str).astype(int)
df['comment'] = df['comment'].astype(str).astype(int)


# In[241]:


df['p_year'] = df['p_year'].astype(str).astype(int)
df['p_month'] = df['p_month'].astype(str).astype(int)
df['p_day'] = df['p_day'].astype(str).astype(int)


# In[244]:


#Fitting a regression model
regr = linear_model.LinearRegression()


# In[270]:


y_train = df['adview']
x_train = df[['views', 'likes', 'dislikes', 'comment', 'category', 'p_year',
       'p_month', 'p_day', 'hr', 'mins', 'sec']]


# In[271]:


regr.fit(x_train, y_train)


# In[272]:


# The coefficients
print('Coefficients: \n', regr.coef_)


# In[273]:


y_pred = regr.predict(x_train)


# In[250]:


y_pred


# In[274]:


print("Mean squared error: %.2f"
      % mean_squared_error(y_train, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# In[254]:


dft = pd.read_csv(r"C:\Users\Piyush\Desktop\machine_learning\ad_org\data\mn\ad_org_test.csv")


# In[255]:


dft = dft.drop(columns = ["vidid"])


# In[257]:


p_year = []
p_month = []
p_day = []

for i in range(0, len(dft)):
    p_year.append(dft['published'][i][0:4])
    p_month.append(dft['published'][i][5:7])
    p_day.append(dft['published'][i][9:])
    
dft['p_year'] = p_year
dft['p_month'] = p_month
dft['p_day'] = p_day


# In[259]:


hr =[]
mins = []
sec = []


# In[260]:


for i in range(0, len(dft)):
    
    temp1 = dft['duration'][i].find('H')
    if(temp1==-1):
        hr.append('0')
        flag = dft['duration'][i].find('M')
        if(flag!=-1):
            temp2 = dft['duration'][i].partition('M')[0]
            mins.append(temp2[2:])
            temp3 = dft['duration'][i].partition('M')[2]
            sec.append(temp3[:-1])
        else:
            mins.append('0')
            temp2 = dft['duration'][i].partition('S')[0]
            sec.append(temp2[2:])
    else:
        temp2 = dft['duration'][i].partition('H')[0]
        hr.append(temp2[2:])
        temp3 = dft['duration'][i].partition('H')[2]
        flag = temp3.find('M')
        if(flag != -1):
            temp4 = temp3.partition('M')[0]
            mins.append(temp4)
            temp5 = temp3.partition('M')[2]
            sec.append(temp5[:-1])
        else:
            mins.append('0')
            temp5 = temp3.partition('S')[0]
            sec.append(temp5)


# In[261]:


dft['hr'] = hr
dft['mins'] = mins
dft['sec'] = sec


# In[262]:


dft = convert_fill(dft)


# In[263]:


le = preprocessing.LabelEncoder()
le.fit(dft['category'])
dft['category'] = le.transform(dft['category'])


# In[264]:


mapping = {'F': -1}
dft = dft.replace({'likes': mapping, 'dislikes': mapping, 'comment': mapping, 'views':mapping})


# In[266]:


dft['views'] = dft['views'].astype(str).astype(int)
dft['likes'] = dft['likes'].astype(str).astype(int)
dft['dislikes'] = dft['dislikes'].astype(str).astype(int)
dft['comment'] = dft['comment'].astype(str).astype(int)


# In[267]:


dft['p_year'] = dft['p_year'].astype(str).astype(int)
dft['p_month'] = dft['p_month'].astype(str).astype(int)
dft['p_day'] = dft['p_day'].astype(str).astype(int)


# In[269]:


df['hr'] = df['hr'].astype(str).astype(int)
df['mins'] = df['mins'].astype(str).astype(int)
df['sec'] = df['sec'].astype(str).astype(int)


# In[268]:


dft.dtypes


# In[275]:


x_train = dft[['views', 'likes', 'dislikes', 'comment', 'category', 'p_year',
       'p_month', 'p_day', 'hr', 'mins', 'sec']]


# In[276]:


y_pred = regr.predict(x_train)


# In[277]:


y_pred

