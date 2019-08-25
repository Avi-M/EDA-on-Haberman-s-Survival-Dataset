
# coding: utf-8

# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

haberman = pd.read_csv("F:\\ADS\\haberman.csv")
print (haberman.shape)


# In[18]:


print(haberman.columns)


# In[19]:


print (haberman.head())


# In[20]:


haberman.head(20)


# In[21]:


haberman.surv_status


# In[22]:


haberman.surv_status.isnull()


# In[24]:


haberman.isnull().sum()


# In[28]:


haberman.plot(kind='scatter', x='axil_nodes', y='Age') ;
plt.grid()
plt.show()


# In[27]:


haberman.columns


# In[30]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="surv_status", size=4)  .map(plt.scatter, "axil_nodes", "Age")  .add_legend();
plt.show();


# In[31]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="surv_status", size=3, vars=['Age','Op_year', 'axil_nodes'])
plt.show()


# In[62]:


import numpy as np
haberman_Long_Survive = haberman.loc[haberman["surv_status"] == 1];
haberman_Short_Survive = haberman.loc[haberman["surv_status"] == 2];
plt.plot(haberman_Long_Survive["axil_nodes"], np.zeros_like(haberman_Long_Survive['axil_nodes']), 'o')
plt.plot(haberman_Short_Survive["axil_nodes"], np.zeros_like(haberman_Short_Survive['axil_nodes']), 'o')
plt.show()


# In[39]:


sns.FacetGrid(haberman, hue="surv_status", size=7)    .map(sns.distplot, "axil_nodes")    .add_legend();
plt.show();


# In[40]:


sns.FacetGrid(haberman, hue="surv_status", size=7)    .map(sns.distplot, "Age")    .add_legend();
plt.show();


# In[41]:


sns.FacetGrid(haberman, hue="surv_status", size=7)    .map(sns.distplot, "Op_year")    .add_legend();
plt.show();


# In[42]:


counts, bin_edges = np.histogram(haberman_Long_Survive['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(haberman_Long_Survive['axil_nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();


# In[43]:


counts, bin_edges = np.histogram(haberman_Long_Survive['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


# In[47]:


counts, bin_edges = np.histogram(haberman_Short_Survive['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(haberman_Short_Survive['axil_nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();


# In[48]:


counts, bin_edges = np.histogram(haberman_Short_Survive['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();


# In[51]:


print("Means:")
print (np.mean(haberman_Long_Survive["axil_nodes"]))
print (np.mean(np.append(haberman_Long_Survive["axil_nodes"],50)))
print (np.mean(haberman_Short_Survive["axil_nodes"]))
print("\nStandard Deviation:")
print(np.mean(haberman_Long_Survive["axil_nodes"]))
print(np.mean(haberman_Short_Survive["axil_nodes"]))


# In[52]:


print("\nMedians:")
print(np.median(haberman_Long_Survive["axil_nodes"]))

print(np.median(np.append(haberman_Long_Survive["axil_nodes"],50)));
print(np.median(haberman_Short_Survive["axil_nodes"]))



print("\nQuantiles:")
print(np.percentile(haberman_Long_Survive["axil_nodes"],np.arange(0, 100, 25)))
print(np.percentile(haberman_Short_Survive["axil_nodes"],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(haberman_Long_Survive["axil_nodes"],90))
print(np.percentile(haberman_Short_Survive["axil_nodes"],90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_Long_Survive["axil_nodes"]))
print(robust.mad(haberman_Short_Survive["axil_nodes"]))


# In[53]:


sns.boxplot(x="surv_status",y="axil_nodes", data=haberman)
plt.show()


# In[56]:


sns.violinplot(x="surv_status",y="axil_nodes", data=haberman, size=8)

plt.show()


# In[60]:


sns.jointplot(x="Age",y="axil_nodes",data=haberman_Long_Survive, kind="kde");
plt.grid();
plt.show();

