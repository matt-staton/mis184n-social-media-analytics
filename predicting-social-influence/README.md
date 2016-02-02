
# Predicting Social Influence

This assignment is based on data from the [Influencers in Social Networks](https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network) competition hosted on Kaggle. 

The dataset comprises a standard, pair-wise preference learning task. Each datapoint describes two individuals using features based on twitter activity (such as volume of interactions, number of followers, etc). The discrete label represents a human judgement about which one of the two individuals is more influential (1 means A > B, 0 means B > A).

The goal of the challenge is to train a machine learning model which, for a pair of individuals, predicts the human judgement on who is more influential with high accuracy. Then, using this model we will quantify the value of influence and explore how a business can identify and leverage influencers.


    import pandas as pd
    import sklearn
    from sklearn.cross_validation import train_test_split
    from sklearn import cross_validation
    from pandas import DataFrame
    import numpy as np
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler
    from numpy import mean
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score


    # Read in data
    data = pd.read_csv("train.csv")


    # Log function
    def transform_features(x):
        return np.log(1+x)


    # Split data into train and test 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(data.ix[:,1:], data.ix[:,0], test_size=0.3, random_state=1)


    # Log transform features
    X_train = transform_features(X_train)
    X_test = transform_features(X_test)


    # Initialize scaler with train
    scaler = StandardScaler()
    scaler.fit(X_train)




    StandardScaler(copy=True, with_mean=True, with_std=True)




    # Scale train and test
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Separate userA and userB data
    X_train_A = X_train_scaled[:,:11]
    X_train_B = X_train_scaled[:,11:]
    X_test_A = X_test_scaled[:,:11]
    X_test_B = X_test_scaled[:,11:]


    # Calculate differences userA - userB for each feature
    X_train_new = X_train_A - X_train_B
    X_test_new = X_test_A - X_test_B


    # Check for multicollinearity
    #DataFrame(X_train_new).corr


    # Initialize models
    models = {'logreg': linear_model.LogisticRegression(C=1.0),
              'boost': GradientBoostingClassifier(n_estimators=100, learning_rate=0.04, random_state=1),
              'rf': RandomForestClassifier(n_estimators=10),
              'knn': KNeighborsClassifier(n_neighbors=5)}


    # Create empty datframe for results
    results_df = DataFrame(columns=['Model', 'Accuracy'])


    # Loop through models and append accuracy scores to results_df
    for k, v in models.iteritems():
        clf = v
        clf.fit(X_train_new, y_train)
        pred = clf.predict(X_test_new)
        results_df = results_df.append({'Model': k, 'Accuracy': accuracy_score(pred, y_test)}, ignore_index=True)


    results_df




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>knn</td>
      <td>0.734545</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rf</td>
      <td>0.727879</td>
    </tr>
    <tr>
      <th>2</th>
      <td>boost</td>
      <td>0.755758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>logreg</td>
      <td>0.755758</td>
    </tr>
  </tbody>
</table>
</div>



Accuracy scores for the 4 models are relatively close. Boosting and logistic regression tie for first with accuracy scores of ~75.58%. A pairwise correlation table of the predictor variables showed multicollinearity between several variables, so we will move forward with the boosting model as it has built-in feature selection.


    # Confusion matrix function
    def accuracy(pred, actual):
        print 'Accuracy: %s' %(np.mean(pred == actual))
        print(pd.crosstab(actual, pred, rownames=['True'], colnames=['Predicted'], margins=True))


    # Run boosting again for confusion matrix
    boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.04, random_state=1)
    boost.fit(X_train_new, y_train)
    boost_pred = clf.predict(X_test_new)


    # Print confusion matrix
    accuracy(boost_pred, y_test)

    Accuracy: 0.755757575758
    Predicted    0    1   All
    True                     
    0          616  219   835
    1          184  631   815
    All        800  850  1650
    


    # Get new feature names
    col_list = []
    for i in list(data.ix[:,1:12].columns.values):
        col_list.append("A-B_"+i[2:])


    # Create feature importance dataframe
    features_df = DataFrame(boost.feature_importances_, index=col_list)
    features_df.columns = ['Importance']


    # Sort dataframe descending by importance
    features_df.sort(['Importance'], ascending=0)




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A-B_listed_count</th>
      <td>0.252978</td>
    </tr>
    <tr>
      <th>A-B_follower_count</th>
      <td>0.181700</td>
    </tr>
    <tr>
      <th>A-B_network_feature_1</th>
      <td>0.178901</td>
    </tr>
    <tr>
      <th>A-B_network_feature_2</th>
      <td>0.093876</td>
    </tr>
    <tr>
      <th>A-B_following_count</th>
      <td>0.085572</td>
    </tr>
    <tr>
      <th>A-B_retweets_received</th>
      <td>0.077644</td>
    </tr>
    <tr>
      <th>A-B_retweets_sent</th>
      <td>0.043102</td>
    </tr>
    <tr>
      <th>A-B_posts</th>
      <td>0.036991</td>
    </tr>
    <tr>
      <th>A-B_mentions_received</th>
      <td>0.023371</td>
    </tr>
    <tr>
      <th>A-B_network_feature_3</th>
      <td>0.013852</td>
    </tr>
    <tr>
      <th>A-B_mentions_sent</th>
      <td>0.012012</td>
    </tr>
  </tbody>
</table>
</div>



Above we have feature importance from the boosting model. Diff in listed_count is the most important predictor of social influence followed by diff in follower_count, diff in network_feature_1, and diff in network_feature_2.


    # Get followers counts for financial value analysis
    fv_df = data.ix[y_test.index,[1,12]]


    # Add actual and predicted values
    fv_df['Actual'] = y_test
    fv_df['Predicted'] = boost_pred


    # Export to csv for analysis
    fv_df.to_csv('financial_value.csv')

Next, we will calculate the financial value of the model, which would be the lift in profits from using analytics versus not.

####Assume a retailer wants influencers to tweet its promotion for a product:
* Without analytics, retailer offers $1 to each person to tweet once
* With analytics, retailer offers $2 to those identified as influencers to send two tweets each
* Non-influencer tweets are no benefit to a retailer
* Influencer tweet leads to a 0.10% chance that a follower will buy one unit of a product
* Influencer tweets leads to a 0.15% chance that a follower will buy one unit of a product
* Retailer profit margin $50 per unit, one customer can buy only one unit

Calculations done in [financial_value.xlsx](https://github.com/juliaawu/mis184n-social-media-analytics/blob/master/predicting-social-influence/financial_value.xlsx)

Given the above assumptions, the promotion effort would have generated $82k in profit with no model, $113k with the boosting model, and $123k with a perfect model. Using analytics to predict social influence would generate a 38% lift in sales, equating to $31k in additional profit.
