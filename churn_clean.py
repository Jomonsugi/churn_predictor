import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from datetime import *
from sklearn.preprocessing import Imputer
from tempfile import TemporaryFile


def read_data(data):
    df = pd.read_csv(data)
    return df

def label_encode(df,encode_list):
    le = LabelEncoder()
    for col in encode_list:
        if df[col].isnull().values.any():
            df[col] = df[col].fillna('')
        le.fit(df[col])
        df[col+'_enc'] = le.transform(df[col])
    return df

def get_dummies(df,get_dummies_list):
    for column in get_dummies_list:
        df = pd.concat([df, pd.get_dummies(df[column])], axis=1)
    return df

def histogram(x, name):
    fig = plt.figure()
    n, bins, patches = plt.hist(x, 200, facecolor='green', alpha=0.75)
    plt.title(r'{}'.format(name))
    # plt.axis([0, 60, 0, 1])
    plt.savefig('hist_{}.png'.format(name))

def avg_surge(df, column):
    df['surge_or_not'] = column
    df['surge_or_not'] = df['surge_or_not'] > 1.00
    df['surge_or_not'] = df['surge_or_not'].astype('int')
    return df

def to_datetime(df):
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    return df

def create_y():
    current_date = datetime.strptime('2014-07-01', '%Y-%m-%d')
    active_date = current_date - timedelta(days=30)
    y = np.array([0 if last_trip_date > active_date else 1 for last_trip_date in df['last_trip_date']])
    return y

def convert_booleans(df):
    '''Convert booleans to 1,0'''
    df['luxury_car_user'] = df['luxury_car_user']*1
    return df

def normalize_inv_hyperbol_sine(df):
    avg_dist = np.array(df['avg_dist'])
    df['avg_dist_norm'] = np.arcsinh(avg_dist)
    trips = np.array(df['trips_in_first_30_days'])
    df['trips_in_first_30_days_norm'] = np.arcsinh(trips)
    return df

def categorize_weekday_pct(df):
    df['all_weekday'] = (df.weekday_pct == 100).astype('int')
    df['all_weekend'] = (df.weekday_pct == 0).astype('int')
    df['mix_weekday_weekend'] = ((df.weekday_pct <100) & (df.weekday_pct > 0)).astype('int')
    df['continuous_weekday_weekend'] = ((df.weekday_pct <100) & (df.weekday_pct > 0))
    return df

def phone(df):
    df['phone'] = df['phone'].apply(lambda x: 0 if x == 'Android' else 1 if x =='iPhone' else x)
    df['phone'] = df['phone'].replace(['nan'], np.nan)
    imp_phone = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp_phone.fit(df[['phone']])
    df['phone'] = imp_phone.transform(df[['phone']]).ravel()
    return df

def avg_rating(df):
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].mean())
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].mean())
    return df

def drop_col(df,columns):
    df = df.drop(columns, axis=1)
    return df


if __name__ == '__main__':
    plt.close('all')
    df = read_data('data/churn_test.csv')

    df = to_datetime(df)
    y = create_y()

    # encode_list = ['phone','city']
    # # df = label_encode(df,encode_list)
    get_dummies_list = ['city']
    df = get_dummies(df,get_dummies_list)
    df = avg_surge(df, df['avg_surge'])
    df = convert_booleans(df)
    df = normalize_inv_hyperbol_sine(df)
    df = categorize_weekday_pct(df)
    df = phone(df)
    df = avg_rating(df)
    drop_col_list = ['avg_dist','avg_surge','city','trips_in_first_30_days','last_trip_date','weekday_pct','signup_date']
    df = drop_col(df,drop_col_list)

    X = preprocessing.StandardScaler().fit_transform(df.astype(float))

    np.save('X_test.npy', X)
    np.save('y_test.npy', y)
