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


def create_features(group):
    group = group.sort_values('purchase_time')

    # Transaction frequency
    group['txn_count_1min'] = group.rolling('1min', on='purchase_time')['purchase_time'].count() - 1
    group['txn_count_10min'] = group.rolling('10min', on='purchase_time')['purchase_time'].count() - 1
    group['txn_count_1h'] = group.rolling('1h', on='purchase_time')['purchase_time'].count() - 1

    # Transaction velocity
    group['amount_sum_1min'] = group.rolling('1min', on='purchase_time')['purchase_value'].sum() - group['purchase_value']
    group['amount_sum_10min'] = group.rolling('10min', on='purchase_time')['purchase_value'].sum() - group['purchase_value']
    group['amount_sum_1h'] = group.rolling('1h', on='purchase_time')['purchase_value'].sum() - group['purchase_value']

    return group


def split_data(df):
   from sklearn.model_selection import train_test_split
   X = df.drop(columns=['class'])  
   y = df['class']  
   xtrain, xtest,ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
   return xtrain, xtest,ytrain,ytest



def encode_fun(df,test,cols):
 import pandas as pd 
 from sklearn.preprocessing import OneHotEncoder
 encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
 encoder.fit(df[cols])
 train_encoded_cols = pd.DataFrame(encoder.transform(df[cols]),columns=encoder.get_feature_names_out(cols),index=df.index)
 test_encoded_cols = pd.DataFrame(encoder.transform(test[cols]),columns=encoder.get_feature_names_out(cols),index=test.index)
 train_df = df.drop(columns=cols)
 test_df = test.drop(columns=cols)
 train_encoded = pd.concat([train_df, train_encoded_cols], axis=1)
 test_encoded = pd.concat([test_df, test_encoded_cols], axis=1)
 return train_encoded, test_encoded


def scale_data(df):
 from sklearn.preprocessing import StandardScaler
 scaler = StandardScaler()
 X_train_scaled = scaler.fit_transform(X_train)
 X_test_scaled = scaler.transform(X_test)
 return  X_train_scaled,  X_test_scaled
 





 
 
