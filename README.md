## Predicting Churn for Ride-Sharing Company

### Research Problem
A ride-sharing company (Company X) is interested in predicting rider retention. Using data for rider activity over a seven month period, develop a model that identifies what factors are best predictors of retention. Also offer suggestions to operationalize insights to help Company X. The model should:
1. Minimize error
2. Facilitate interpretation of factors that contribute to the predictions.

### Data
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
weekday_pct: the percent of the user’s trips occurring during a weekday```

### Exploratory Data Analysis

We discovered that some of the predictor variables (e.g., average distance, number of trips in first 30 days) were positively skewed to a rather marked degree. These variables also included zero values so it was not possible to use simple corrections for skew, such as log transform.
![][https://user-images.githubusercontent.com/17363251/27752602-111d0122-5d9f-11e7-9dc0-d2fce2363102.png]

We normalized skewed data using an inverse hyperbolic sine transformation.

![][https://user-images.githubusercontent.com/17363251/27753738-0f4f5da4-5da4-11e7-8066-dac9a9af3307.png]

While examining distributions of the variables, we noticed that the percent of users' trips occurring during a weekday had an interesting distribution, with definite spikes for 0% and 100% and a more normal looking distribution for the space between 0 and 100.

![][https://user-images.githubusercontent.com/17363251/27754012-5bf7456c-5da5-11e7-9a41-dff6fd296075.png]

We decided to create dummy variables to split this variable apart:
    1. All rides on weekdays
    2. All rides on weekends
    3. Mix of weekdays and weekends


```
def categorize_weekday_pct(df):
    df['all_weekday'] = (df.weekday_pct == 100).astype('int')
    df['all_weekend'] = (df.weekday_pct == 0).astype('int')
    df['mix_weekday_weekend'] = ((df.weekday_pct <100) & (df.weekday_pct > 0)).astype('int')
```
