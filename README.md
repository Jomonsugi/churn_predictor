## Predicting Churn for Ride-Sharing Company

### Research Problem
A ride-sharing company (Company X) is interested in predicting rider retention. Using data for rider activity over a seven month period, develop a model that identifies what factors are best predictors of retention. Also offer suggestions to operationalize insights to help Company X. The model should:
1. Minimize error
2. Facilitate interpretation of factors that contribute to the predictions.

### Data

We have a mix of rider demographics, rider behavior, ride characteristics, and rider/driver ratings of each other.
```
city: city this user signed up in
phone: primary device for this user
signup_date: date of account registration; in the form `YYYYMMDD`
last_trip_date: the last time this user completed a trip; in the form `YYYYMMDD`
avg_dist: the average distance (in miles) per trip taken in the first 30 days after signup
avg_rating_by_driver: the rider’s average rating over all of their trips
avg_rating_of_driver: the rider’s average rating of their drivers over all of their trips
surge_pct: the percent of trips taken with surge multiplier > 1
avg_surge: The average surge multiplier over all of this user’s trips
trips_in_first_30_days: the number of trips this user took in the first 30 days after signing up
luxury_car_user: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise
weekday_pct: the percent of the user’s trips occurring during a weekday
```
Data were provided in csv files, so it was simple to read them into Pandas dataframes:
```python
df_train = pd.read_csv('data/churn_train.csv')
df_test = pd.read_csv('data/churn_test.csv')
```

#### Defining Churn

Next, we needed to pull dates out of timestamps to  calculate the churn outcome variable. Users were identified as having churned if they had not used the ride-share service in the past thirty days. This function converts timestamps to date time objects and calculates our churn outcome variable:

```python
def convert_dates(df):
    df['last_trip_date'] = df['last_trip_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df['signup_date'] = df['signup_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    current_date = datetime.strptime('2014-07-01', '%Y-%m-%d')
    active_date = current_date - timedelta(days=30)
    y = np.array([0 if last_trip_date > active_date else 1 for last_trip_date in df['last_trip_date']])
    return y
```

Categorical variables where classes were represented with strings were encoded as numerical classes:

```python
def label_encode(df, encode_list):
    le = preprocessing.LabelEncoder()
    for col in encode_list:
        le.fit(df[col])
        df[col + '_enc'] = le.transform(df[col])
    return df
```


### Exploratory Data Analysis

We discovered that some of the predictor variables (e.g., average distance, number of trips in first 30 days) were positively skewed to a rather marked degree. These variables also included zero values so it was not possible to use simple corrections for skew, such as log transform.

![](https://user-images.githubusercontent.com/17363251/27752602-111d0122-5d9f-11e7-9dc0-d2fce2363102.png)

Skewed data were normalized using an inverse hyperbolic sine transformation:

```python
def normalize_inv_hyperbol_sine(x):
    x_arr = np.array(df[x])
    df[x+'_normalized'] = np.arcsinh(x_arr)
```
This worked well to normalize the data:

![](https://user-images.githubusercontent.com/17363251/27753738-0f4f5da4-5da4-11e7-8066-dac9a9af3307.png)

While examining distributions of the variables, we noticed that the percent of users' trips occurring during a weekday had an interesting distribution, with definite spikes for 0% and 100% and a more normal/Gaussian-looking distribution for the space between 0 and 100:

![](https://user-images.githubusercontent.com/17363251/27754012-5bf7456c-5da5-11e7-9a41-dff6fd296075.png)

We decided to create dummy variables to split this variable apart:
1. All rides on weekdays
2. All rides on weekends
3. Mix of weekdays and weekends


```python
def categorize_weekday_pct(df):
    df['all_weekday'] = (df.weekday_pct == 100).astype('int')
    df['all_weekend'] = (df.weekday_pct == 0).astype('int')
    df['mix_weekday_weekend'] = ((df.weekday_pct <100) & (df.weekday_pct > 0)).astype('int')
```

### Contributors
Our team included Micah Shanks ([github.com/Jomonsugi](https://github.com/Jomonsugi)), Stuart King ([github.com/Stuart-D-King](https://github.com/Stuart-D-King), Jennifer Waller ([github.com/jw15](https://github.com/jw15)), and Ian
