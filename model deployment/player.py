#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import pickle
import pandas as pd
#mat plot
import numpy as np
#Sk learn imports
from sklearn import tree,preprocessing
#ensembles
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
import sklearn.metrics as metrics
#scores
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score,auc  
#models
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score,learning_curve,GridSearchCV,validation_curve
from sklearn import tree,preprocessing


df = pd.read_csv(r"C:\model deployment\players.csv")


# In[42]:


def drop_columns(df):
    df.drop(df.loc[:, :'Name' ],axis=1, inplace = True)
    df.drop(df.loc[:, 'Photo':'Special'],axis=1, inplace = True)
    df.drop(df.loc[:, 'International Reputation':'Real Face' ],axis=1, inplace = True)
    df.drop(df.loc[:, 'Jersey Number':'Contract Valid Until' ],axis=1, inplace = True)
    df.drop(df.loc[:, 'LS':'RB'],axis=1, inplace = True)
    df.drop(df.loc[:, 'GKDiving':'Release Clause'],axis=1, inplace = True)
    
def weight_to_int(df):
    df['Weight'] = df['Weight'].str[:-3]
    df['Weight'] = df['Weight'].apply(lambda x: int(x))
    return df
def impute_data(df):
    df.dropna(inplace=True)


# In[43]:


drop_columns(df)
impute_data(df)


# In[44]:


weight_to_int(df)


# In[45]:


def height_convert(df_height):
        try:
            feet = int(df_height[0])
            dlm = df_height[-2]
            if dlm == "'":
                height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)
            elif dlm != "'":
                height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)
        except ValueError:
            height = 0
        return height
def height_to_int(df):
    df['Height'] = df['Height'].apply(height_convert)

height_to_int(df)


# In[46]:


#Transform positions to 3 categories 'Striker', 'Midfielder', 'Defender' 
def transform_positions(df):
    for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:
      df.loc[df.Position == i , 'Position'] = 'Striker' 
    
    for i in ['CAM', 'CDM', 'LCM', 'CM', 'LAM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM']:
      df.loc[df.Position == i , 'Position'] = 'Midfielder' 
    
    for i in ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB','GK']:
      df.loc[df.Position == i , 'Position'] = 'Defender' 
transform_positions(df)    


# In[47]:


df1=df.iloc[:,1:11]
df1


# In[48]:


encoder = preprocessing.LabelEncoder()
df1['Preferred Foot']=encoder.fit_transform(df1['Preferred Foot'].values)
encoder.classes_


# In[49]:


positions = df1["Position"].unique()
encoder = preprocessing.LabelEncoder()
df1['Position'] = encoder.fit_transform(df['Position'])
encoder.classes_


# In[50]:


#Target variable
Y=df1['Position']
#The other features are all but the position
df1.drop(columns=["Position"],inplace=True)
#Split the data
X_train, X_test, y_train, y_test = train_test_split(df1,Y, test_size=0.30, 
                                                    random_state=42)


# In[51]:


gridsearch_forest = RandomForestClassifier()
params = {
    "n_estimators": [1, 10, 100],
    "max_depth": [5,8,15],
    "min_samples_leaf" : [1, 2, 4]}
RF = GridSearchCV(gridsearch_forest, param_grid=params, cv=5 )


# In[53]:


model=RF.fit(X_train,y_train)


# In[56]:


pickle.dump(model, open('player.pkl', 'wb'))


# In[ ]:




