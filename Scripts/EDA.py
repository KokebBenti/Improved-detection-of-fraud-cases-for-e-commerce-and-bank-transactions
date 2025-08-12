def duplicate_check(df):
 import pandas as pd
 df=df.drop_duplicates()
 return df

def change_type(df,col):
 import pandas as pd
 df[col]=pd.to_datetime(df[col])
 return df   

def plot_numerical(df,col):
 import matplotlib.pyplot as plt
 df[col].hist(bins=50, figsize=(8,6))
 plt.title(col)
 plt.xlim(df[col].min(), df[col].max())
 plt.xlabel(col)
 plt.ylabel("Freq")
 plt.show()

def plot_categorical(df,col):
 import matplotlib.pyplot as plt 
 counts = df[col].value_counts()
 counts.plot(kind='bar')
 plt.title(col)
 plt.xlabel('Category')
 plt.ylabel('Frequency')
 plt.show()   

def correlation(df):
 num_df = df.select_dtypes(include=['int64', 'float64']) 
 m=num_df.corr()  
 return m

def box_plot(df,col):
 import matplotlib.pyplot as plt
 plt.boxplot(df[col])
 plt.title(col)
 plt.show() 


def remove_outliers(df,col):
 from scipy.stats import zscore
 df['z_score'] = zscore(df[col])
 df=df[abs(df['z_score']) < 3]
 return df


def to_int(df,col):
 df[col] = df[col].astype(int)  
 return df
