# Predicting Churn for A Ride-Sharing Company

## Research Problem
A ride-sharing company (Company X) is interested in predicting rider retention. Using data for rider activity, we developed a model that identifies what factors are best predictors of retention. We also offer suggestions to operationalize insights to help Company X.

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

<!-- Data  in csv files, so it was simple to read into Pandas dataframes:

```python
df_train = pd.read_csv('data/churn_train.csv')
df_test = pd.read_csv('data/churn_test.csv')
``` -->

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


## Exploratory Data Analysis and Feature Engineering

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

## Classification/Predictive Analytics

Random Forest is a great place to start with a classification problem like this. It's fast, easy to use, and pretty accurate right out of the box. Our Random Forest Classifier produced an F1 Score of 77% on unseen data.

To improve our model fit, we next tried some boosted classification models. While boosted models require more tuning (and therefore take a bit longer to get working than Random Forest), they are usually more accurate than Random Forest.
1. Gradient boost  
  - Using Scikit Learn's `GridSearchCV`, we first performed a grid search to determine the best model parameters for a `GradientBoostingClassifier`. The resultant classifier performed well, with an F1 Score of 83% on unseen data.
2. XGBoost
  - XGBoost did a good job as well with near equal results on the unseen data. The average F1 score from cross validation results was almost 84%.

## Results

Accuracy, recall, and precision on unseen data that XGBoost produced confirmed that it is a good choice as it generalizes well for this application.

- Accuracy: 78.29%
- Recall: 86.25%
- Precision: 80.74%

Although there could be possible improvements from further feature engineering, the current model would certainly be helpful in identifying customer segments that should be further investigated.

By running a feature importance analysis on the XGBoost model, it is seems that surge percentage, average distance of ride, and number of trips taken in the first 30 days are all the most relevant predictive features in this model. Next steps would include comparing those who are predicted to churn and those who are not against these three features. This could lead to actionable insights and thus would be a priority of continuing work on this project.

<!-- I ran feature importance on the XGBoost and found that surge_pct, avg_dist_norm, and trips_in_first_30_days are of most importance, but the why would take more project time that I do not have currently have. If I can find some time at any point, I think the only proper way to make any recommendations to company x would be to look at the distributions of those predicted to churn and those who are not against these three variables and then test for a statistical difference from there -->

## Recommendations for Company X

* Use the best fitting model (above) to obtain predicted probabilities for individuals. Target those with greater than some probability of churning (choose this cutoff by considering profit curve based on confusion matrix).

* Further investigate the variables stated above that are important predictors of churn

* Offer discounts or free rides to at-risk users to try and retain them - no need to target users below a certain probability threshold.


## What We Learned

### How useful is feature engineering and normalizing skewed data?

Classifiers like random forest and boosted trees are quite robust to skewed and non-normally distributed data. We probably did not need to spend time transforming our data or creating dummy variables for percent of weekday rides.

<!-- However, some of the feature engineering seemed to yield improvements in model fit. Specifically, ... WHAT ENGINEERED FEATURES WERE USEFUL? -->

## Contributors
Our team included Micah Shanks ([github.com/Jomonsugi](https://github.com/Jomonsugi)), Stuart King ([github.com/Stuart-D-King](https://github.com/Stuart-D-King)), Jennifer Waller ([github.com/jw15](https://github.com/jw15)), and Ian

## Tech Stack

![](https://user-images.githubusercontent.com/17363251/27755513-51b86baa-5dad-11e7-81eb-b3b59f0f8b0a.png)
