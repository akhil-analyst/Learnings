
# coding: utf-8

# In[1]:


#one way to load iris_Dataset

from sklearn.datasets import load_iris

iris = load_iris()
iris


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import seaborn as sns

'''downlaod iris.csv from https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'''

iris_df= pd.read_csv('/Users/a.k/Documents/Python/iris/iris.csv')


# In[16]:


iris_df.head()


# In[17]:


iris_df.describe()


# In[20]:


iris_df.shape


# In[21]:


iris_df.columns


# In[22]:


iris_df['species'].value_counts()


# In[ ]:


##balanced data set


# In[24]:


iris_df.plot(x='sepal_length',y='sepal_width',kind='scatter',grid='True') ;
plt.show()


# In[30]:


sns.set_style("whitegrid");
sns.FacetGrid(iris_df,hue='species',size=6)    .map(plt.scatter,"sepal_length","sepal_width")    .add_legend();
plt.show()


# In[ ]:


#setosa is easily istinguishable


# In[32]:


sns.pairplot(iris_df,hue='species')


# In[ ]:


##petal length and petal width are most important features in indentifying the categories
# setosa can be easily identified , while there is an overlap b/w vericolor and virginica


# In[39]:


sns.FacetGrid(iris_df,hue = "species",size = 6)    .map(sns.distplot,"petal_length")    .add_legend();
plt.show()


# In[42]:


iris_df_setosa = iris_df.loc[iris_df["species"] == "setosa"];
iris_df_versicolor = iris_df.loc[iris_df["species"] == "versicolor"];
iris_df_virginica = iris_df.loc[iris_df["species"] == "virginica"];

plt.plot(iris_df_setosa["petal_length"],np.zeros_like(iris_df_setosa["petal_length"]),'o')
plt.plot(iris_df_versicolor["petal_length"],np.zeros_like(iris_df_versicolor["petal_length"]),'o')
plt.plot(iris_df_virginica["petal_length"],np.zeros_like(iris_df_virginica["petal_length"]),'o')
plt.show()


# In[43]:


##cumulative distribution

counts, bin_edges = np.histogram(iris_df_setosa['petal_length'], bins=10, 
                                 density = True)


# In[53]:


print(counts)
print(bin_edges)
print(bin_edges[1:])


# In[54]:


pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:],cdf);


# In[55]:


counts, bin_edges = np.histogram(iris_df_setosa['petal_length'], bins=10, 
                                 density = True)

pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:],cdf);


counts, bin_edges = np.histogram(iris_df_versicolor['petal_length'], bins=10, 
                                 density = True)

pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:],cdf);


counts, bin_edges = np.histogram(iris_df_virginica['petal_length'], bins=10, 
                                 density = True)

pdf = counts/sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:],cdf);

plt.show()


# In[58]:


print(np.percentile(iris_df_setosa['petal_length'],np.arange(0,100,25)))


# In[63]:


print(np.mean(iris_df_setosa['petal_length']))
print(np.median(iris_df_setosa['petal_length']))
print(np.std(iris_df_setosa['petal_length']))


# In[60]:


from statsmodels import robust
print(robust.mad(iris_df_setosa['petal_length']))


# In[67]:


sns.boxplot(x = "species" ,y = "petal_length", data =iris_df)


# In[70]:


#best of both box plot and pdf functions
sns.violinplot(x='species',y="petal_length",data=iris_df)


# In[ ]:


#multivariate probability. density contour plot


# In[73]:


sns.jointplot(x="petal_length",y="petal_width",data=iris_df_setosa,kind="kde")
plt.show()

