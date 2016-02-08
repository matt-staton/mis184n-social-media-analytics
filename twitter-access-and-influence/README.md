
# Twitter Access & Influence

In the following, we will write a script to collect tweets from the Twitter API about the upcoming iPhone 7. The goal is to identify users who are most influential on the topic using the importance scores we got from the gradient boosting model in the [Predicting Social Influence](https://github.com/juliaawu/mis184n-social-media-analytics/tree/master/predicting-social-influence) assignment. Then, we will create a network of tweets and retweets about the iPhone 7.


    import oauth2
    import time
    import urllib2
    import json
    import pandas as pd
    import re
    import sklearn
    from sklearn import preprocessing


    # Fixed authentication parameters for Twitter API
    url1 = "https://api.twitter.com/1.1/search/tweets.json"
    params = {
        "oauth_version": "1.0",
        "oauth_nonce": oauth2.generate_nonce(),
        "oauth_timestamp": int(time.time())
    }


    # Variable authentication parameters
    api_key = 'HQPqqS0OeXv1F5kVEIpCjAfJV'
    api_secret = 'p8fkGtt5mcgXDqR827qtDMdjc85cupfSP88ay3cGHyCnIVCQ0G'
    access_token = '224485242-3X6fKZK2fvMwkcRn8OqQGc0EbJK9oRaOUv2irTtY'
    access_secret = 'VHSm5OmkNR1a7WCxMjCKQxpeHfnhMjqz84r0jU7r780IZ'
    
    consumer = oauth2.Consumer(key=api_key, secret=api_secret)
    token = oauth2.Token(key=access_token, secret=access_secret)
    
    params["oauth_consumer_key"] = consumer.key
    params["oauth_token"] = token.key


    # Search tweets for keyword "iPhone 7"
    maxID = -1
    search_results_final = []
    
    for i in range(65):
        url = url1
        params["q"] = "iPhone 7"
        params["count"] = 100
        params["lang"] = 'en'
        params['max_id'] = maxID
        req = oauth2.Request(method="GET", url=url, parameters=params)
        signature_method = oauth2.SignatureMethod_HMAC_SHA1()
        req.sign_request(signature_method, consumer, token)
        url = req.to_url()
        response = urllib2.Request(url)
        search_results = json.load(urllib2.urlopen(response))
        for i in search_results['statuses']:
            maxID = int(i['id_str'])-1
            search_results_final.append(i)


    # Retrieve user info, tweet, and retweet count
    user_information = []
    tweets = []
    retweet_count = []
    
    for i in search_results_final:
        i['user']['entities'] = ''
        user_information.append(i['user'])
        tweets.append(i['text'])
        retweet_count.append(i['retweet_count'])


    # Save in df
    tweets_df =pd.DataFrame(user_information)
    tweets_df['text'] = tweets
    tweets_df['retweet_count'] = retweet_count

Now that we have the data, we can calculate the scores. We will use the top 4 important attributes as indicated by the boosting model, with the exception of network feature 1 and 2 since they are not a part of this dataset.


    # Get columns for calculating score
    score = tweets_df[['screen_name', 'listed_count', 'followers_count', 'friends_count', 'retweet_count']]


    # Normalize columns
    cols_to_norm = ['listed_count', 'followers_count', 'friends_count', 'retweet_count']
    score[cols_to_norm] = score[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

For the weights, we will use the importance score for that attribute / the sum of the other importance scores.


    # Calculate score using importance scores from gradient boosting model in Predicting Social Influence as weights
    score['score'] = 0.423115134*score['listed_count'] + 0.303900022*score['followers_count'] + 0.143122359*score['friends_count'] + 0.129862484*score['retweet_count']

    C:\Users\Julia Wu\Anaconda\lib\site-packages\IPython\kernel\__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from IPython.kernel.zmq import kernelapp as app
    


    # Sort descending by score to get top 20 influential users
    score.sort('score', ascending=0).head(20)




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>listed_count</th>
      <th>followers_count</th>
      <th>friends_count</th>
      <th>retweet_count</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1543</th>
      <td>TheNextWeb</td>
      <td>0.993889</td>
      <td>0.558258</td>
      <td>0.001409</td>
      <td>-0.062545</td>
      <td>0.582264</td>
    </tr>
    <tr>
      <th>886</th>
      <td>TeenVogue</td>
      <td>0.298730</td>
      <td>0.868529</td>
      <td>0.012519</td>
      <td>-0.062740</td>
      <td>0.383987</td>
    </tr>
    <tr>
      <th>2914</th>
      <td>abpnewstv</td>
      <td>0.074455</td>
      <td>0.998537</td>
      <td>-0.005915</td>
      <td>-0.063175</td>
      <td>0.325908</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>macworld</td>
      <td>0.460977</td>
      <td>0.165527</td>
      <td>-0.002704</td>
      <td>-0.062871</td>
      <td>0.236799</td>
    </tr>
    <tr>
      <th>2139</th>
      <td>scarletmonahan</td>
      <td>0.020404</td>
      <td>0.590566</td>
      <td>0.164909</td>
      <td>-0.063240</td>
      <td>0.203496</td>
    </tr>
    <tr>
      <th>1478</th>
      <td>applenws</td>
      <td>0.200966</td>
      <td>0.347932</td>
      <td>-0.006379</td>
      <td>-0.062437</td>
      <td>0.181747</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>intlspectator</td>
      <td>0.055694</td>
      <td>0.170748</td>
      <td>0.621862</td>
      <td>-0.061395</td>
      <td>0.156485</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>phone_crazy</td>
      <td>0.008950</td>
      <td>0.042664</td>
      <td>0.993368</td>
      <td>-0.063261</td>
      <td>0.150710</td>
    </tr>
    <tr>
      <th>2953</th>
      <td>phone_crazy</td>
      <td>0.008950</td>
      <td>0.042664</td>
      <td>0.993368</td>
      <td>-0.063261</td>
      <td>0.150710</td>
    </tr>
    <tr>
      <th>2318</th>
      <td>leyahdeshon</td>
      <td>-0.006015</td>
      <td>-0.001348</td>
      <td>-0.002888</td>
      <td>0.936739</td>
      <td>0.118279</td>
    </tr>
    <tr>
      <th>854</th>
      <td>BrooksLaloni</td>
      <td>-0.006047</td>
      <td>-0.001334</td>
      <td>-0.003332</td>
      <td>0.936739</td>
      <td>0.118207</td>
    </tr>
    <tr>
      <th>2393</th>
      <td>camillelaurenn_</td>
      <td>-0.006047</td>
      <td>-0.001296</td>
      <td>-0.003455</td>
      <td>0.936739</td>
      <td>0.118200</td>
    </tr>
    <tr>
      <th>2467</th>
      <td>LOWKEYDAVEE</td>
      <td>-0.006047</td>
      <td>-0.001375</td>
      <td>-0.003961</td>
      <td>0.936739</td>
      <td>0.118104</td>
    </tr>
    <tr>
      <th>2202</th>
      <td>yooitvalerieeee</td>
      <td>-0.006111</td>
      <td>-0.001386</td>
      <td>-0.004132</td>
      <td>0.936739</td>
      <td>0.118049</td>
    </tr>
    <tr>
      <th>3729</th>
      <td>belin_isabell</td>
      <td>-0.006111</td>
      <td>-0.001324</td>
      <td>-0.004309</td>
      <td>0.936739</td>
      <td>0.118043</td>
    </tr>
    <tr>
      <th>634</th>
      <td>miaacaylenn</td>
      <td>-0.006111</td>
      <td>-0.001392</td>
      <td>-0.004446</td>
      <td>0.936739</td>
      <td>0.118003</td>
    </tr>
    <tr>
      <th>2116</th>
      <td>megannespinosaa</td>
      <td>-0.006111</td>
      <td>-0.001407</td>
      <td>-0.004439</td>
      <td>0.936739</td>
      <td>0.117999</td>
    </tr>
    <tr>
      <th>4873</th>
      <td>DrewJanssen3529</td>
      <td>-0.006111</td>
      <td>-0.001386</td>
      <td>-0.004624</td>
      <td>0.936739</td>
      <td>0.117979</td>
    </tr>
    <tr>
      <th>2370</th>
      <td>janelledestiny</td>
      <td>-0.006111</td>
      <td>-0.001320</td>
      <td>-0.005013</td>
      <td>0.936739</td>
      <td>0.117943</td>
    </tr>
    <tr>
      <th>2346</th>
      <td>gennnnesiis</td>
      <td>-0.006079</td>
      <td>-0.001326</td>
      <td>-0.005225</td>
      <td>0.936739</td>
      <td>0.117925</td>
    </tr>
  </tbody>
</table>
</div>



Above we have the most influential users on the topic of iPhone 7.

Next, we will format the data so that we can build a network of tweets and retweets on the topic. To do this, we need to extract user2 from the text.


    # Set user1 as screen_name
    user1 = [('@'+x) for x in tweets_df['screen_name']]
    user2 = []
    tweet_type = []


    # Extract user2 from text. User2 should be the user who retweeted or replied, tweet_type indicates Tweet or RT
    for index, value in enumerate(tweets_df['text']):
        if value[:2] == 'RT':
            text = value[3:].split()[0][:-1]
            user2.append(text)
            tweet_type.append('RT')
        elif value[:1] == '@':
            text = value[0:].split()[0]
            user2.append(text)
            tweet_type.append('RT')
        else:
            user2.append('@'+tweets_df.ix[index,'screen_name'])
            tweet_type.append('Tweet')    


    network = pd.concat([pd.Series(user1, name='user1'), pd.Series(user2, name='user2'), pd.Series(tweet_type, name='tweet_type')], axis=1)
    network.to_csv('network.csv', index=False)

Now we can use the csv file to produce a network graph in Gephi. Here is the [graph](https://github.com/juliaawu/mis184n-social-media-analytics/blob/master/twitter-access-and-influence/network.PNG) with a filter for degree range > 2.
