# Predicting Churn for Ride-Sharing Company

## Research Problem
A ride-sharing company (Company X) is interested in predicting rider retention. Using data for rider activity, develop a model that identifies what factors are best predictors of retention. Also offer suggestions to operationalize insights to help Company X. The model should:
1. Minimize error
2. Facilitate interpretation of factors that contribute to the predictions.

## Data

We have a mix of rider demographics, rider behavior, ride characteristics, and rider/driver ratings of each other. Data spanned a 7 month period.

Variable     | Description                |
-------------| ----------------------- |
city | City this user signed up in
phone | Primary device for this user
signup_date |Date of account registration
last_trip_date | Last time user completed a trip
avg_dist | Average distance (in miles) per trip taken in first 30 days after signup
avg_rating_by_driver | Rider’s average rating over all trips
avg_rating_of_driver | Rider’s average rating of their drivers over all trips
surge_pct | Percent of trips taken with surge multiplier > 1
avg_surge | Average surge multiplier over all of user’s trips
trips_in_first_30_days | Number of trips user took in first 30 days after signing up
luxury_car_user | TRUE if user took luxury car in first 30 days
weekday_pct | Percent of user’s trips occurring during a weekday

Data were provided in csv files, so it was simple to read into Pandas dataframes:

```python
df_train = pd.read_csv('data/churn_train.csv')
df_test = pd.read_csv('data/churn_test.csv')
```

### Defining Churn

We converted dates into date time objects to calculate the churn outcome variable. Users were identified as having churned if they had not used the ride-share service in the past thirty days:

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


## Exploratory Data Analysis

We discovered that some of the predictor variables (e.g., average distance, number of trips in first 30 days) were positively skewed to a rather marked degree. These variables also included zero values so it was not possible to use simple corrections for skew, such as log transform.

![](https://user-images.githubusercontent.com/17363251/27752602-111d0122-5d9f-11e7-9dc0-d2fce2363102.png)

Skewed data were normalized using an inverse hyperbolic sine transformation:

```python
def normalize_inv_hyperbol_sine(x):
    x_arr = np.array(df[x])
    df[x+'_normalized'] = np.arcsinh(x_arr)
```
This worked well to normalize the data.

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

## Predictive Analytics

Random Forest is a great place to start with a classification problem like this. It's fast, easy to use, and pretty accurate right out of the box.

To improve our model fit, we next tried some boosted classification models. While boosted models require more tuning (and therefore take a bit longer to get working than Random Forest), they are usually more accurate than Random Forest.
1. Gradient boost
2. XGBoost

## What We Learned

### How useful is feature engineering and normalizing skewed data?

## Contributors
Our team included Micah Shanks ([github.com/Jomonsugi](https://github.com/Jomonsugi)), Stuart King ([github.com/Stuart-D-King](https://github.com/Stuart-D-King), Jennifer Waller ([github.com/jw15](https://github.com/jw15)), and Ian
