#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation
# ## Auther: Suchismita Mallick
# ### DataScience and Business Analytics Intern 
# ### Task 1 : Prediction Using Supervise Machine Learning

# In[1]:


import pandas as pd #use for read our file or data
import numpy as np   #use for matrix multiplication
import matplotlib.pyplot as plt # data Vidualization
import seaborn as sns #plotting a figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ## Import Data

# In[2]:


A=pd.read_csv("/Users/Admin/Desktop/Task1 grip.csv")


# In[3]:


A.head()


# In[4]:


# Check if there any null value in the Dataset
A.isnull == True


# In[5]:


A.shape


# In[6]:


A.describe()


# In[7]:


A.info()


# ## Detect Outliers from the given Dataset

# In[8]:


H=A['Hours']
S=A['Scores']


# In[9]:


sns.boxplot(H)


# In[10]:


sns.boxplot(S)


# ##### From the above boxplots we see that there is no outliers present in our dataset.

# ## Scatter Plots of the data

# In[11]:


sns.set_style('darkgrid')
A.plot(kind='scatter',x='Hours',y='Scores', color='green',marker='*',);
plt.title('Scores Vs Study Hours',size=20)


# In[12]:


# From the above scatter plot there looks to be correlation between the 'Marks Percentage' and 'Hours Studied', Lets plot a regression line to confirm the correlation.
sns.regplot(x= A['Hours'], y= A['Scores'], color='green', marker='*')
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(A.corr())


# ###### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and % of score.

# In[13]:


## Defining X and y from the Data
X = A.iloc[:, :-1].values  
y = A.iloc[:, 1].values


# ### Splitting the Dataset in Testing And Training

# In[14]:


train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)


# ### Now Fitting The Dataset Into The Model

# In[15]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("Trained the Model")


# In[16]:


## Making Prediction


# In[17]:


pred_y = regression.predict(test_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in test_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# ### Comparing Actual vs Prediction

# In[18]:


compare_scores = pd.DataFrame({'Actual Marks': test_y, 'Predicted Marks': pred_y})
compare_scores


# In[19]:


plt.scatter(x=test_X, y=test_y, color='green', marker='*')
plt.plot(test_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.xlabel('Hours Studied', size=12)
plt.ylabel('Marks Percentage', size=12)
plt.show()


# In[20]:


## Now we have to calculating the accuracy of the model


# In[22]:


print('Mean absolute error: ',mean_absolute_error(test_y,pred_y))


# ###### If a student studies for 9.25hrs/day then what will be the predicted scores?

# In[26]:


Hours = [9.25]
answer = regression.predict([Hours])
print("Score = {}".format(round(answer[0],3)))


# ### According to this Regression Model if a student studies 9.25hrs/day she/he is likely to score 93.89 marks.

# In[ ]:




