
# Web Crawling & Brand Associations

In the following, we will develop a web crawler to fetch posts from Edmunds.com's discussion forums. We will then use that data to understand associations between car brands and associations between car brands and their attributes.

Below is a web crawler that collects 5,000+ posts from the [Entry Level Luxury Performance Sedans](http://forums.edmunds.com/discussion/2864/general/x/entry-level-luxury-performance-sedans) forum on Edmunds.com.

```python
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re


# Initialize dataframe and lists to store fetched posts
edmunds = DataFrame()
username = []
date = []
message = []


# Crawl the forum and extract username, date, and post
page = 1
while page < 200:
	r = requests.get('http://forums.edmunds.com/discussion/2864/general/x/entry-level-luxury-performance-sedans/p' + str(page))
	soup = BeautifulSoup(r.text)
	username.extend([x.get_text() for x in soup.find_all("a", attrs={"class": "Username"})])
	date.extend([x['datetime'][:10] for x in soup.find_all("time")])
	message.extend([x.get_text().strip() for x in soup.find_all("div", attrs={"class": "Message"})])
	page += 1


# Store data into dataframe
edmunds['username'] = username
edmunds['date'] = date
edmunds['message'] = message
```
## Associations Between Car Brands

To allow for a higher-level car brand analysis, we will clean the data by searching for model names and replacing them with brands.

```python
# Read in lookup table of brands and models
brands = pd.read_csv("brand_lookup.csv")


# Create a dictionary to use for search and replace
brands_df = brands.ix[:,'Replace']
brands_df.index = brands.ix[:,'Search']
brands_dict = brands_df.to_dict()


# Replace NaN posts with an empty string
edmunds['message'] = edmunds['message'].replace(np.nan,' ', regex=True)


# Lowercase all text
edmunds['message'] = edmunds['message'].str.lower()


# Find models and replace with brands
pattern = re.compile(r'\b(' + '|'.join(brands_dict.keys()) + r')\b')
for index, message in enumerate(edmunds['message']):
	edmunds['message'][index] = pattern.sub(lambda x: brands_dict[x.group()], message)


# Create binary document term matrix
countvec = CountVectorizer(binary=True, stop_words='english')
DTM = pd.DataFrame(countvec.fit_transform(edmunds['message']).toarray(), columns=countvec.get_feature_names())


# Sum the DTM to get word counts
word_counts = DTM.sum()


# Sort word counts descending
word_counts.sort(ascending=0)


# View word counts to identify top brands
word_counts.head(20)
```



    car            3029
    bmw            1754
    like           1524
    just           1397
    acura          1350
    infiniti       1219
    don            1172
    think          1125
    drive           893
    sedan           882
    better          869
    performance     823
    new             796
    know            758
    people          752
    good            748
    really          735
    best            718
    driving         701
    want            661
    dtype: int64



```python
# Initialize list of top 10 brands and create top brands DTM
top10_brands = ['bmw', 'acura', 'infiniti', 'lexus', 'audi', 'cadillac', 'honda', 'nissan', 'toyota', 'mercedes']
top10_DTM = DTM.ix[:, top10_brands]


# Create dictionary of brand counts
brand_count = {}
for brand in top10_brands:
	brand_count[brand] = word_counts.ix[brand]
```
We will use lift ratios as a measure of brand association. Lift ratios tell us whether words appear together by chance or due to association. The formula for calculating lift is P(A & B) = P(A & B) / (P(A) * P(B)), where P() indicates the probability of. A lift ratio of < 1 indicates that the words are less likely to appear together than by chance, while a lift ratio of > 1 indicates association between two words. The higher the number, the greater the association.

```python
# Initialize lists to hold combination of brands and their lift scores
combo_list = []
lift_list = []


# Loop through top 10 car brands and calculate lift scores for each brand combination
for i in xrange(0, 10):
	for j in xrange(1, 10):
		if j > i:
			combo_list.append(top10_brands[i] + " & " + top10_brands[j])
			combo_count = sum(top10_DTM[top10_brands[i]] + top10_DTM[top10_brands[j]] == 2)
			n = len(top10_DTM)
			lift = float(combo_count * n) / (brand_count[top10_brands[i]] * brand_count[top10_brands[j]])
			lift_list.append(lift)


# Store into dataframe
lift_df = DataFrame()
lift_df['brands'] = combo_list
lift_df['brand1'] = [x[0:re.search('&', x).start()-1] for x in lift_df['brands']]
lift_df['brand2'] = [x[re.search('&', x).start()+2:] for x in lift_df['brands']]
lift_df['lift'] = lift_list


# View lift scores
lift_df
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brands</th>
      <th>brand1</th>
      <th>brand2</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bmw &amp; acura</td>
      <td>bmw</td>
      <td>acura</td>
      <td>1.240649</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bmw &amp; infiniti</td>
      <td>bmw</td>
      <td>infiniti</td>
      <td>1.552704</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bmw &amp; lexus</td>
      <td>bmw</td>
      <td>lexus</td>
      <td>2.143397</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bmw &amp; audi</td>
      <td>bmw</td>
      <td>audi</td>
      <td>1.785820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bmw &amp; cadillac</td>
      <td>bmw</td>
      <td>cadillac</td>
      <td>1.451411</td>
    </tr>
    <tr>
      <th>5</th>
      <td>bmw &amp; honda</td>
      <td>bmw</td>
      <td>honda</td>
      <td>1.158504</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bmw &amp; nissan</td>
      <td>bmw</td>
      <td>nissan</td>
      <td>1.157830</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bmw &amp; toyota</td>
      <td>bmw</td>
      <td>toyota</td>
      <td>1.323863</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bmw &amp; mercedes</td>
      <td>bmw</td>
      <td>mercedes</td>
      <td>1.805268</td>
    </tr>
    <tr>
      <th>9</th>
      <td>acura &amp; infiniti</td>
      <td>acura</td>
      <td>infiniti</td>
      <td>1.966568</td>
    </tr>
    <tr>
      <th>10</th>
      <td>acura &amp; lexus</td>
      <td>acura</td>
      <td>lexus</td>
      <td>1.575729</td>
    </tr>
    <tr>
      <th>11</th>
      <td>acura &amp; audi</td>
      <td>acura</td>
      <td>audi</td>
      <td>1.522659</td>
    </tr>
    <tr>
      <th>12</th>
      <td>acura &amp; cadillac</td>
      <td>acura</td>
      <td>cadillac</td>
      <td>1.634325</td>
    </tr>
    <tr>
      <th>13</th>
      <td>acura &amp; honda</td>
      <td>acura</td>
      <td>honda</td>
      <td>2.223060</td>
    </tr>
    <tr>
      <th>14</th>
      <td>acura &amp; nissan</td>
      <td>acura</td>
      <td>nissan</td>
      <td>1.697183</td>
    </tr>
    <tr>
      <th>15</th>
      <td>acura &amp; toyota</td>
      <td>acura</td>
      <td>toyota</td>
      <td>1.228601</td>
    </tr>
    <tr>
      <th>16</th>
      <td>acura &amp; mercedes</td>
      <td>acura</td>
      <td>mercedes</td>
      <td>1.440814</td>
    </tr>
    <tr>
      <th>17</th>
      <td>infiniti &amp; lexus</td>
      <td>infiniti</td>
      <td>lexus</td>
      <td>2.012872</td>
    </tr>
    <tr>
      <th>18</th>
      <td>infiniti &amp; audi</td>
      <td>infiniti</td>
      <td>audi</td>
      <td>1.766592</td>
    </tr>
    <tr>
      <th>19</th>
      <td>infiniti &amp; cadillac</td>
      <td>infiniti</td>
      <td>cadillac</td>
      <td>1.797301</td>
    </tr>
    <tr>
      <th>20</th>
      <td>infiniti &amp; honda</td>
      <td>infiniti</td>
      <td>honda</td>
      <td>1.423321</td>
    </tr>
    <tr>
      <th>21</th>
      <td>infiniti &amp; nissan</td>
      <td>infiniti</td>
      <td>nissan</td>
      <td>2.221312</td>
    </tr>
    <tr>
      <th>22</th>
      <td>infiniti &amp; toyota</td>
      <td>infiniti</td>
      <td>toyota</td>
      <td>1.122522</td>
    </tr>
    <tr>
      <th>23</th>
      <td>infiniti &amp; mercedes</td>
      <td>infiniti</td>
      <td>mercedes</td>
      <td>1.577097</td>
    </tr>
    <tr>
      <th>24</th>
      <td>lexus &amp; audi</td>
      <td>lexus</td>
      <td>audi</td>
      <td>2.205919</td>
    </tr>
    <tr>
      <th>25</th>
      <td>lexus &amp; cadillac</td>
      <td>lexus</td>
      <td>cadillac</td>
      <td>1.850384</td>
    </tr>
    <tr>
      <th>26</th>
      <td>lexus &amp; honda</td>
      <td>lexus</td>
      <td>honda</td>
      <td>1.433521</td>
    </tr>
    <tr>
      <th>27</th>
      <td>lexus &amp; nissan</td>
      <td>lexus</td>
      <td>nissan</td>
      <td>1.224519</td>
    </tr>
    <tr>
      <th>28</th>
      <td>lexus &amp; toyota</td>
      <td>lexus</td>
      <td>toyota</td>
      <td>2.852109</td>
    </tr>
    <tr>
      <th>29</th>
      <td>lexus &amp; mercedes</td>
      <td>lexus</td>
      <td>mercedes</td>
      <td>3.430509</td>
    </tr>
    <tr>
      <th>30</th>
      <td>audi &amp; cadillac</td>
      <td>audi</td>
      <td>cadillac</td>
      <td>2.304505</td>
    </tr>
    <tr>
      <th>31</th>
      <td>audi &amp; honda</td>
      <td>audi</td>
      <td>honda</td>
      <td>1.281220</td>
    </tr>
    <tr>
      <th>32</th>
      <td>audi &amp; nissan</td>
      <td>audi</td>
      <td>nissan</td>
      <td>1.359517</td>
    </tr>
    <tr>
      <th>33</th>
      <td>audi &amp; toyota</td>
      <td>audi</td>
      <td>toyota</td>
      <td>1.132931</td>
    </tr>
    <tr>
      <th>34</th>
      <td>audi &amp; mercedes</td>
      <td>audi</td>
      <td>mercedes</td>
      <td>3.419392</td>
    </tr>
    <tr>
      <th>35</th>
      <td>cadillac &amp; honda</td>
      <td>cadillac</td>
      <td>honda</td>
      <td>0.969358</td>
    </tr>
    <tr>
      <th>36</th>
      <td>cadillac &amp; nissan</td>
      <td>cadillac</td>
      <td>nissan</td>
      <td>0.986735</td>
    </tr>
    <tr>
      <th>37</th>
      <td>cadillac &amp; toyota</td>
      <td>cadillac</td>
      <td>toyota</td>
      <td>0.857163</td>
    </tr>
    <tr>
      <th>38</th>
      <td>cadillac &amp; mercedes</td>
      <td>cadillac</td>
      <td>mercedes</td>
      <td>2.571490</td>
    </tr>
    <tr>
      <th>39</th>
      <td>honda &amp; nissan</td>
      <td>honda</td>
      <td>nissan</td>
      <td>4.498425</td>
    </tr>
    <tr>
      <th>40</th>
      <td>honda &amp; toyota</td>
      <td>honda</td>
      <td>toyota</td>
      <td>5.753036</td>
    </tr>
    <tr>
      <th>41</th>
      <td>honda &amp; mercedes</td>
      <td>honda</td>
      <td>mercedes</td>
      <td>1.657822</td>
    </tr>
    <tr>
      <th>42</th>
      <td>nissan &amp; toyota</td>
      <td>nissan</td>
      <td>toyota</td>
      <td>4.701005</td>
    </tr>
    <tr>
      <th>43</th>
      <td>nissan &amp; mercedes</td>
      <td>nissan</td>
      <td>mercedes</td>
      <td>1.577960</td>
    </tr>
    <tr>
      <th>44</th>
      <td>toyota &amp; mercedes</td>
      <td>toyota</td>
      <td>mercedes</td>
      <td>2.198916</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Save to csv
lift_df.to_csv(r'lift.csv', index=False)
```
To visualize the data, we can make a network diagram using a tool like [Gephi](https://gephi.org/) or a multi-dimensional scaling map using [XLSTAT](https://www.xlstat.com/en/download) with Excel.

* The [first network graph](https://github.com/juliaawu/mis184n-social-media-analytics/blob/master/web-crawling-and-brand-associations/network_graph.PNG) doesn't tell us much. All brands are talked about in association with each other with lift ratios > 1. However, [filtering the graph by lifts > 3](https://github.com/juliaawu/mis184n-social-media-analytics/blob/master/web-crawling-and-brand-associations/network_graph_filtered.PNG) reveals some interesting brand associations. Nissan, Honda, and Toyota are all linked, which indicates that consumers may find these brands similar. This makes sense because the brands are all economical. We also see Mercedes linked to Audi, Cadillac, and Lexus, which suggest that consumers may view Mercedes to be a point of comparison for luxury vehicles. Acura is linked to Honda, Cadillac, and Lexus in a similar way, which implies that it could be a focal point for mid-range cars. BMW and Infiniti have no connections with lift > 3. This might be because they are more unqiue in nature. Acura, Cadillac, and Audi also seem to be bridges in terms of association between a lower-end to higher-end vehicles.


* The [MDS graph](https://github.com/juliaawu/mis184n-social-media-analytics/blob/master/web-crawling-and-brand-associations/mds.PNG) shows Toyota, Honda, and Nissan clustered together in the bottom left, indicating their similarity. This is reflective of what we saw in the network graph. Mercedes, Audi, Cadillac, and Lexus are also close together with Mercedes in the center as the focal point. Acura and Infiniti are in the lower right quadrant and BMW is at the top of the graph by itself. The axis of MDS graphs are open to interpretation. Looking at the placement of these brands, I would infer the x-axis to be an indication of price and the y-axis to be an indication of performance.

This type of analysis would be useful for car companies in evaluating consumer perception and identifying direct competitors.

##Associations Between Car Brands and Attributes
Now we will conduct a similar analysis between the top 5 car brands and car attributes to identify which attributes are most commonly talked about for each brand.

```python
# Read in lookup table of attributes and their synonyms
attributes = pd.read_csv("attribute_lookup.csv")


# Create a dictionary to use for search and replace
attributes_df = attributes.ix[:,'Replace']
attributes_df.index = attributes.ix[:,'Search']
attributes_dict = attributes_df.to_dict()


# Find attribute synonyms and replace with attribute
pattern = re.compile(r'\b(' + '|'.join(attributes_dict.keys()) + r')\b')
for index, message in enumerate(edmunds['message']):
	edmunds['message'][index] = pattern.sub(lambda x: attributes_dict[x.group()], message)


# Create binary document term matrix
DTM2 = pd.DataFrame(countvec.fit_transform(edmunds['message']).toarray(), columns=countvec.get_feature_names())


# Sum the DTM to get word counts
word_counts2 = DTM2.sum()


# Sort word counts descending
word_counts2.sort(ascending=0)


# View word counts to identify top attributes
word_counts2.head(25)
```



    car             3029
    performance     2434
    bmw             1754
    economy         1629
    like            1524
    just            1397
    acura           1350
    infiniti        1219
    don             1172
    engine          1140
    think           1125
    drive            893
    sedan            882
    better           869
    design           800
    new              796
    know             758
    people           752
    good             748
    really           735
    best             718
    does             633
    say              620
    way              616
    aspirational     611
    dtype: int64



The top 5 most talked about attributes are performance, economy, engine, design, and aspirational.

```python
# Initialize list of top attributes and intitalize variable of top 5 brands
atts = [x for x in attributes['Replace'].unique() if x in DTM2.columns]
top5_brands = top10_brands[0:5]


# Create top 5 brands and attributes DTMs
top5_brands_DTM = DTM.ix[:,top10_brands[0:5]]
atts_DTM = DTM.ix[:,atts]


# Create dictionary of attribute counts
att_count = {}
for att in atts:
	att_count[att] = word_counts2.ix[att]


# Initialize lists to hold combination of brands and attributes and their lift scores
combo_list2 = []
lift_list2 = []


# Loop through top 5 car brands and attributes and calculate lift scores for each brand + attribute combination
for i in xrange(0, 5):
	for j in xrange(0, len(atts)):
		combo_list2.append(top5_brands[i] + " & " + atts[j])
		combo_count = sum(top5_brands_DTM[top5_brands[i]] + atts_DTM[atts[j]] == 2)
		n = len(top5_brands_DTM)
		lift = float(combo_count * n) / (brand_count[top5_brands[i]] * att_count[atts[j]])
		lift_list2.append(lift)


# Store into dataframe
lift_df2 = DataFrame()
lift_df2['combo'] = combo_list2
lift_df2['brand'] = [x[0:re.search('&', x).start()-1] for x in lift_df2['combo']]
lift_df2['attribute'] = [x[re.search('&', x).start()+2:] for x in lift_df2['combo']]
lift_df2['lift'] = lift_list2


# View lift scores
lift_df2.sort(columns=['brand','lift'], ascending=False)
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>combo</th>
      <th>brand</th>
      <th>attribute</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90</th>
      <td>lexus &amp; hybrid</td>
      <td>lexus</td>
      <td>hybrid</td>
      <td>3.008818</td>
    </tr>
    <tr>
      <th>98</th>
      <td>lexus &amp; service</td>
      <td>lexus</td>
      <td>service</td>
      <td>2.606650</td>
    </tr>
    <tr>
      <th>95</th>
      <td>lexus &amp; reliability</td>
      <td>lexus</td>
      <td>reliability</td>
      <td>1.878116</td>
    </tr>
    <tr>
      <th>91</th>
      <td>lexus &amp; interior</td>
      <td>lexus</td>
      <td>interior</td>
      <td>1.685696</td>
    </tr>
    <tr>
      <th>94</th>
      <td>lexus &amp; price</td>
      <td>lexus</td>
      <td>price</td>
      <td>1.531762</td>
    </tr>
    <tr>
      <th>80</th>
      <td>lexus &amp; brand</td>
      <td>lexus</td>
      <td>brand</td>
      <td>1.360237</td>
    </tr>
    <tr>
      <th>101</th>
      <td>lexus &amp; warranty</td>
      <td>lexus</td>
      <td>warranty</td>
      <td>1.358821</td>
    </tr>
    <tr>
      <th>81</th>
      <td>lexus &amp; dealer</td>
      <td>lexus</td>
      <td>dealer</td>
      <td>1.253674</td>
    </tr>
    <tr>
      <th>100</th>
      <td>lexus &amp; transmission</td>
      <td>lexus</td>
      <td>transmission</td>
      <td>1.141042</td>
    </tr>
    <tr>
      <th>86</th>
      <td>lexus &amp; experience</td>
      <td>lexus</td>
      <td>experience</td>
      <td>1.055940</td>
    </tr>
    <tr>
      <th>92</th>
      <td>lexus &amp; noise</td>
      <td>lexus</td>
      <td>noise</td>
      <td>1.012583</td>
    </tr>
    <tr>
      <th>88</th>
      <td>lexus &amp; features</td>
      <td>lexus</td>
      <td>features</td>
      <td>0.888542</td>
    </tr>
    <tr>
      <th>85</th>
      <td>lexus &amp; engine</td>
      <td>lexus</td>
      <td>engine</td>
      <td>0.757483</td>
    </tr>
    <tr>
      <th>93</th>
      <td>lexus &amp; performance</td>
      <td>lexus</td>
      <td>performance</td>
      <td>0.532168</td>
    </tr>
    <tr>
      <th>96</th>
      <td>lexus &amp; safety</td>
      <td>lexus</td>
      <td>safety</td>
      <td>0.491440</td>
    </tr>
    <tr>
      <th>99</th>
      <td>lexus &amp; size</td>
      <td>lexus</td>
      <td>size</td>
      <td>0.394086</td>
    </tr>
    <tr>
      <th>87</th>
      <td>lexus &amp; exterior</td>
      <td>lexus</td>
      <td>exterior</td>
      <td>0.337984</td>
    </tr>
    <tr>
      <th>102</th>
      <td>lexus &amp; wheel</td>
      <td>lexus</td>
      <td>wheel</td>
      <td>0.318581</td>
    </tr>
    <tr>
      <th>79</th>
      <td>lexus &amp; design</td>
      <td>lexus</td>
      <td>design</td>
      <td>0.302762</td>
    </tr>
    <tr>
      <th>97</th>
      <td>lexus &amp; seating</td>
      <td>lexus</td>
      <td>seating</td>
      <td>0.093193</td>
    </tr>
    <tr>
      <th>83</th>
      <td>lexus &amp; economy</td>
      <td>lexus</td>
      <td>economy</td>
      <td>0.058182</td>
    </tr>
    <tr>
      <th>78</th>
      <td>lexus &amp; availability</td>
      <td>lexus</td>
      <td>availability</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>82</th>
      <td>lexus &amp; ease</td>
      <td>lexus</td>
      <td>ease</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>84</th>
      <td>lexus &amp; efficiency</td>
      <td>lexus</td>
      <td>efficiency</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>89</th>
      <td>lexus &amp; functionality</td>
      <td>lexus</td>
      <td>functionality</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>103</th>
      <td>lexus &amp; aspirational</td>
      <td>lexus</td>
      <td>aspirational</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>infiniti &amp; noise</td>
      <td>infiniti</td>
      <td>noise</td>
      <td>1.883953</td>
    </tr>
    <tr>
      <th>68</th>
      <td>infiniti &amp; price</td>
      <td>infiniti</td>
      <td>price</td>
      <td>1.692132</td>
    </tr>
    <tr>
      <th>65</th>
      <td>infiniti &amp; interior</td>
      <td>infiniti</td>
      <td>interior</td>
      <td>1.576963</td>
    </tr>
    <tr>
      <th>72</th>
      <td>infiniti &amp; service</td>
      <td>infiniti</td>
      <td>service</td>
      <td>1.551929</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>audi &amp; functionality</td>
      <td>audi</td>
      <td>functionality</td>
      <td>0.059432</td>
    </tr>
    <tr>
      <th>104</th>
      <td>audi &amp; availability</td>
      <td>audi</td>
      <td>availability</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>108</th>
      <td>audi &amp; ease</td>
      <td>audi</td>
      <td>ease</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>129</th>
      <td>audi &amp; aspirational</td>
      <td>audi</td>
      <td>aspirational</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>acura &amp; price</td>
      <td>acura</td>
      <td>price</td>
      <td>1.511849</td>
    </tr>
    <tr>
      <th>39</th>
      <td>acura &amp; interior</td>
      <td>acura</td>
      <td>interior</td>
      <td>1.376210</td>
    </tr>
    <tr>
      <th>40</th>
      <td>acura &amp; noise</td>
      <td>acura</td>
      <td>noise</td>
      <td>1.275855</td>
    </tr>
    <tr>
      <th>28</th>
      <td>acura &amp; brand</td>
      <td>acura</td>
      <td>brand</td>
      <td>1.013596</td>
    </tr>
    <tr>
      <th>43</th>
      <td>acura &amp; reliability</td>
      <td>acura</td>
      <td>reliability</td>
      <td>0.957839</td>
    </tr>
    <tr>
      <th>36</th>
      <td>acura &amp; features</td>
      <td>acura</td>
      <td>features</td>
      <td>0.926058</td>
    </tr>
    <tr>
      <th>38</th>
      <td>acura &amp; hybrid</td>
      <td>acura</td>
      <td>hybrid</td>
      <td>0.842469</td>
    </tr>
    <tr>
      <th>46</th>
      <td>acura &amp; service</td>
      <td>acura</td>
      <td>service</td>
      <td>0.832043</td>
    </tr>
    <tr>
      <th>44</th>
      <td>acura &amp; safety</td>
      <td>acura</td>
      <td>safety</td>
      <td>0.796133</td>
    </tr>
    <tr>
      <th>34</th>
      <td>acura &amp; experience</td>
      <td>acura</td>
      <td>experience</td>
      <td>0.767126</td>
    </tr>
    <tr>
      <th>29</th>
      <td>acura &amp; dealer</td>
      <td>acura</td>
      <td>dealer</td>
      <td>0.722116</td>
    </tr>
    <tr>
      <th>47</th>
      <td>acura &amp; size</td>
      <td>acura</td>
      <td>size</td>
      <td>0.616088</td>
    </tr>
    <tr>
      <th>33</th>
      <td>acura &amp; engine</td>
      <td>acura</td>
      <td>engine</td>
      <td>0.550930</td>
    </tr>
    <tr>
      <th>50</th>
      <td>acura &amp; wheel</td>
      <td>acura</td>
      <td>wheel</td>
      <td>0.520349</td>
    </tr>
    <tr>
      <th>49</th>
      <td>acura &amp; warranty</td>
      <td>acura</td>
      <td>warranty</td>
      <td>0.475587</td>
    </tr>
    <tr>
      <th>41</th>
      <td>acura &amp; performance</td>
      <td>acura</td>
      <td>performance</td>
      <td>0.459741</td>
    </tr>
    <tr>
      <th>48</th>
      <td>acura &amp; transmission</td>
      <td>acura</td>
      <td>transmission</td>
      <td>0.459269</td>
    </tr>
    <tr>
      <th>35</th>
      <td>acura &amp; exterior</td>
      <td>acura</td>
      <td>exterior</td>
      <td>0.403446</td>
    </tr>
    <tr>
      <th>27</th>
      <td>acura &amp; design</td>
      <td>acura</td>
      <td>design</td>
      <td>0.254320</td>
    </tr>
    <tr>
      <th>45</th>
      <td>acura &amp; seating</td>
      <td>acura</td>
      <td>seating</td>
      <td>0.097853</td>
    </tr>
    <tr>
      <th>37</th>
      <td>acura &amp; functionality</td>
      <td>acura</td>
      <td>functionality</td>
      <td>0.072508</td>
    </tr>
    <tr>
      <th>31</th>
      <td>acura &amp; economy</td>
      <td>acura</td>
      <td>economy</td>
      <td>0.046157</td>
    </tr>
    <tr>
      <th>30</th>
      <td>acura &amp; ease</td>
      <td>acura</td>
      <td>ease</td>
      <td>0.028053</td>
    </tr>
    <tr>
      <th>32</th>
      <td>acura &amp; efficiency</td>
      <td>acura</td>
      <td>efficiency</td>
      <td>0.027701</td>
    </tr>
    <tr>
      <th>26</th>
      <td>acura &amp; availability</td>
      <td>acura</td>
      <td>availability</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51</th>
      <td>acura &amp; aspirational</td>
      <td>acura</td>
      <td>aspirational</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>130 rows × 4 columns</p>
</div>



```python
# Filter by brand
lift_df2[lift_df2['brand'] == 'audi'].sort(columns=['lift'], ascending=False)
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>combo</th>
      <th>brand</th>
      <th>attribute</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>124</th>
      <td>audi &amp; service</td>
      <td>audi</td>
      <td>service</td>
      <td>2.261375</td>
    </tr>
    <tr>
      <th>120</th>
      <td>audi &amp; price</td>
      <td>audi</td>
      <td>price</td>
      <td>1.957705</td>
    </tr>
    <tr>
      <th>117</th>
      <td>audi &amp; interior</td>
      <td>audi</td>
      <td>interior</td>
      <td>1.819210</td>
    </tr>
    <tr>
      <th>118</th>
      <td>audi &amp; noise</td>
      <td>audi</td>
      <td>noise</td>
      <td>1.673252</td>
    </tr>
    <tr>
      <th>121</th>
      <td>audi &amp; reliability</td>
      <td>audi</td>
      <td>reliability</td>
      <td>1.662594</td>
    </tr>
    <tr>
      <th>127</th>
      <td>audi &amp; warranty</td>
      <td>audi</td>
      <td>warranty</td>
      <td>1.637268</td>
    </tr>
    <tr>
      <th>106</th>
      <td>audi &amp; brand</td>
      <td>audi</td>
      <td>brand</td>
      <td>1.359517</td>
    </tr>
    <tr>
      <th>112</th>
      <td>audi &amp; experience</td>
      <td>audi</td>
      <td>experience</td>
      <td>1.296884</td>
    </tr>
    <tr>
      <th>107</th>
      <td>audi &amp; dealer</td>
      <td>audi</td>
      <td>dealer</td>
      <td>0.887848</td>
    </tr>
    <tr>
      <th>122</th>
      <td>audi &amp; safety</td>
      <td>audi</td>
      <td>safety</td>
      <td>0.870091</td>
    </tr>
    <tr>
      <th>126</th>
      <td>audi &amp; transmission</td>
      <td>audi</td>
      <td>transmission</td>
      <td>0.859289</td>
    </tr>
    <tr>
      <th>111</th>
      <td>audi &amp; engine</td>
      <td>audi</td>
      <td>engine</td>
      <td>0.830021</td>
    </tr>
    <tr>
      <th>114</th>
      <td>audi &amp; features</td>
      <td>audi</td>
      <td>features</td>
      <td>0.679759</td>
    </tr>
    <tr>
      <th>125</th>
      <td>audi &amp; size</td>
      <td>audi</td>
      <td>size</td>
      <td>0.678345</td>
    </tr>
    <tr>
      <th>128</th>
      <td>audi &amp; wheel</td>
      <td>audi</td>
      <td>wheel</td>
      <td>0.603214</td>
    </tr>
    <tr>
      <th>116</th>
      <td>audi &amp; hybrid</td>
      <td>audi</td>
      <td>hybrid</td>
      <td>0.517911</td>
    </tr>
    <tr>
      <th>119</th>
      <td>audi &amp; performance</td>
      <td>audi</td>
      <td>performance</td>
      <td>0.446842</td>
    </tr>
    <tr>
      <th>105</th>
      <td>audi &amp; design</td>
      <td>audi</td>
      <td>design</td>
      <td>0.421450</td>
    </tr>
    <tr>
      <th>113</th>
      <td>audi &amp; exterior</td>
      <td>audi</td>
      <td>exterior</td>
      <td>0.385809</td>
    </tr>
    <tr>
      <th>123</th>
      <td>audi &amp; seating</td>
      <td>audi</td>
      <td>seating</td>
      <td>0.120311</td>
    </tr>
    <tr>
      <th>110</th>
      <td>audi &amp; efficiency</td>
      <td>audi</td>
      <td>efficiency</td>
      <td>0.113530</td>
    </tr>
    <tr>
      <th>109</th>
      <td>audi &amp; economy</td>
      <td>audi</td>
      <td>economy</td>
      <td>0.106825</td>
    </tr>
    <tr>
      <th>115</th>
      <td>audi &amp; functionality</td>
      <td>audi</td>
      <td>functionality</td>
      <td>0.059432</td>
    </tr>
    <tr>
      <th>104</th>
      <td>audi &amp; availability</td>
      <td>audi</td>
      <td>availability</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>108</th>
      <td>audi &amp; ease</td>
      <td>audi</td>
      <td>ease</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>129</th>
      <td>audi &amp; aspirational</td>
      <td>audi</td>
      <td>aspirational</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



The brand + attribute lift dataframe is sorted by brand and lift so that we can see each brand with their top attributes listed first. For example, we see that Lexus is most talked about for its hybrid, service and reliability, Acura is mentioned for its price and interior, and Audi is known for its service.

```python
lift_df2[lift_df2['attribute'] == "aspirational"].sort(columns="lift", ascending=False)
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>combo</th>
      <th>brand</th>
      <th>attribute</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>infiniti &amp; aspirational</td>
      <td>infiniti</td>
      <td>aspirational</td>
      <td>1.475095</td>
    </tr>
    <tr>
      <th>8</th>
      <td>acura &amp; aspirational</td>
      <td>acura</td>
      <td>aspirational</td>
      <td>1.418823</td>
    </tr>
    <tr>
      <th>18</th>
      <td>lexus &amp; aspirational</td>
      <td>lexus</td>
      <td>aspirational</td>
      <td>1.396072</td>
    </tr>
    <tr>
      <th>23</th>
      <td>audi &amp; aspirational</td>
      <td>audi</td>
      <td>aspirational</td>
      <td>1.388443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bmw &amp; aspirational</td>
      <td>bmw</td>
      <td>aspirational</td>
      <td>1.376174</td>
    </tr>
  </tbody>
</table>
</div>



The data also indicate that Infiniti is the most aspirational brand of the 5.
