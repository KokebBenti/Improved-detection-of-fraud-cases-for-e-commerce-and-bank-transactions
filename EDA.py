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

def box_plot(df,col):
 import matplotlib.pyplot as plt
 plt.boxplot(df[col])
 plt.title(col)
 plt.show() 