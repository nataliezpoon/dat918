# w09d01: Natural Language Processing (NLP)

#### Learning Objectives
- Discuss the major tasks involved with NLP.
- Discuss, on a low level, the components of NLP.
- Identify why NLP is difficult.
- Demonstrate <u>text classification</u>.
- Demonstrate common <u>text processing techniques</u>.

### How Do We Use NLP in Data Science?

In data science, we are often asked to analyze unstructured text or make a predictive model using it. Unfortunately, most data science techniques require numeric data. 
<font color=yellow>NLP libraries provide a tool set of methods to convert unstructured text into meaningful numeric data.</font>

- **ANALYSIS:** NLP techniques provide tools to allow us to understand and analyze large amounts of text. 
- For example:
    - Analyze the positivity/negativity of comments on different websites. 
    - Extract key words from meeting notes and visualize how meeting topics change over time 
      - (*more explortory data analysis*)

- **VECTORIZING FOR MEACHINE LEARNING:** When building a machine learning model, we typically <u>must transform our data into numeric features.</u> This process of transforming non-numeric data such as natural language into numeric features is called <font color=yellow>vectorization</font>. 
- For example:
    - Understanding related words using <font color=yellow>stemming</font>, NLP lets us know that "swim", "swims", and "swimming" all refer to the same base word. This allows us to reduce the number of features used in our model.
    - Identifying important and unique words using <font color=yellow>TF-IDF (term frequency-inverse document frequency)</font>, we can identify which words are most likely to be meaningful in a document.
      - converts documents into vectors

---- 
### Install TextBlob

The TextBlob Python library provides a simplified interface for exploring common NLP tasks including part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

To proceed with the lesson, first install TextBlob, as explained below. We tend to prefer Anaconda-based installations, since they tend to be tested with our other Anaconda packages.

**To install textblob run:**
`conda install -c https://conda.anaconda.org/sloria textblob`
**Or:**
`pip install textblob`
`python -m textblob.download_corpora lite`

### Lesson Guide
- [Introduction to Natural Language Processing](#intro)
- [Reading Yelp reviews With NLP](#yelp_rev)
- [Text Classification](#text_class)
- [Count Vectorization](#count_vec)
    - [Using CountVectorizer in a Model](#countvectorizer-model)
    - [N-Grams](#ngrams)
    - [Stop-Word Removal](#stopwords)
	- [Count Vector Options](#cvec_opt)
- [Intro to TextBlob](#textblob)
	- [Stemming and Lemmatization](#stem)
- [Term Frequency–Inverse Document Frequency Vectorization](#tfidf)
	- [Yelp Summary Using TF–IDF](#yelp_tfidf)
- [Sentiment Analysis](#sentiment)
- [BONUS: Adding Features to a Document-Term Matrix](#add_feat)
- [BONUS: More TextBlob Features](#more_textblob)
- [APPENDIX: Intro to Naive Bayes and Text Classification](#bayes)
- [Conclusion](#conclusion)

----
*Adapted from [NLP Crash Course](http://files.meetup.com/7616132/DC-NLP-2013-09%20Charlie%20Greenbacker.pdf) by Charlie Greenbacker and [Introduction to NLP](http://spark-public.s3.amazonaws.com/nlp/slides/intro.pdf) by Dan Jurafsky*

### What Is Natural Language Processing (NLP)?

- Using computers to process (analyze, understand, generate) natural human languages.
- Making sense of human knowledge stored as unstructured text.
- Building probabilistic models using data about a language.

### What Are Some of the Higher-Level Task Areas?

- **Objective:** Discuss the major tasks involved with natural language processing.

We often hope that computers can solve many high-level problems involving natural language. Unfortunately, due to the difficulty of understanding human language, many of these problems are still not well solved. That said, existing solutions to these problems all involve utilizing the lower-level components of NLP discussed in the next section. Some higher-level tasks include:

- **Chatbots:** Understand natural language from the user and return intelligent responses.
    - [Api.ai](https://api.ai/)
- **Information retrieval:** Find relevant results and similar results.
    - [Google](https://www.google.com/)    
- **Information extraction:** Structured information from unstructured documents.
    - [Events from Gmail](https://support.google.com/calendar/answer/6084018?hl=en)
- **Machine translation:** One language to another.
    - [Google Translate](https://translate.google.com/)
- **Text simplification:** Preserve the meaning of text, but simplify the grammar and vocabulary.
    - [Rewordify](https://rewordify.com/)
    - [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page)
- **Predictive text input:** Faster or easier typing.
    - [Phrase completion application](https://justmarkham.shinyapps.io/textprediction/)
    - [A much better application](https://farsite.shinyapps.io/swiftkey-cap/)
- **Sentiment analysis:** Attitude of speaker.
    - [Hater News](https://medium.com/@KevinMcAlear/building-hater-news-62062c58325c)
- **Automatic summarization:** Extractive or abstractive summarization.
    - [autotldr](https://www.reddit.com/r/technology/comments/35brc8/21_million_people_still_use_aol_dialup/cr2zzj0)
- **Natural language generation:** Generate text from data.
    - [How a computer describes a sports match](http://www.bbc.com/news/technology-34204052)
    - [Publishers withdraw more than 120 gibberish papers](http://www.nature.com/news/publishers-withdraw-more-than-120-gibberish-papers-1.14763)
- **Speech recognition and generation:** Speech-to-text, text-to-speech.
    - [Google's Web Speech API demo](https://www.google.com/intl/en/chrome/demos/speech.html)
    - [Vocalware Text-to-Speech demo](https://www.vocalware.com/index/demo)
- **Question answering:** Determine the intent of the question, match query with knowledge base, evaluate hypotheses.
    - [How did supercomputer Watson beat Jeopardy champion Ken Jennings?](http://blog.ted.com/how-did-supercomputer-watson-beat-jeopardy-champion-ken-jennings-experts-discuss/)
    - [IBM's Watson Trivia Challenge](http://www.nytimes.com/interactive/2010/06/16/magazine/watson-trivia-game.html)
    - [The AI Behind Watson](http://www.aaai.org/Magazine/Watson/watson.php)

### What Are Some of the Lower-Level Components?

- **Objective:** Discuss, on a low level, the components of natural language processing.

Unfortunately, the NLP programming libraries typically do not provide direct solutions for the high-level tasks above. Instead, they provide low-level building blocks that enable us to craft our own solutions. These include:

- **Tokenization:** Breaking text into tokens (words, sentences, n-grams)
- **Stop-word removal:** a/an/the
- **Stemming and lemmatization:** root word
- **TF-IDF:** word importance
- **Part-of-speech tagging:** noun/verb/adjective
- **Named entity recognition:** person/organization/location
- **Spelling correction:** "New Yrok City"
- **Word sense disambiguation:** "buy a mouse"
- **Segmentation:** "New York City subway"
- **Language detection:** "translate this page"
- **Machine learning:** specialized models that work well with text

### Why is NLP hard?

- **Objective:** Identify why natural language processing is difficult.

Natural language processing requires an understanding of the language and the world. Several limitations of NLP are:

- **Ambiguity**:
    - Hospitals Are Sued by 7 Foot Doctors
    - Juvenile Court to Try Shooting Defendant
    - Local High School Dropouts Cut in Half
- **Non-standard English:** text messages
- **Idioms:** "throw in the towel"
- **Newly coined words:** "retweet"
- **Tricky entity names:** "Where is A Bug's Life playing?"
- **World knowledge:** "Mary and Sue are sisters", "Mary and Sue are mothers"
----
## Exercise: Reading in the Yelp Reviews

Throughout this lesson, we will use Yelp reviews to practice and discover common low-level NLP techniques.

#### Frequently Used NLP Terms
- <font color=yellow>**corpus**</font>: a collection of documents (derived from the Latin word for "body")
- <font color=yellow>**corpora**</font>: plural form of corpus

Throughout this lesson, we will use a model very popular for text classification called <font color=yellow>**Naive Bayes**</font> (the "NB" in `BinonmialNB` and `MultinomialNB` below). 
- it works exactly the same as all other models in scikit-learn! 
- see the [appendix](#bayes) at the end of this notebook for a quick introduction.
```python
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer

%matplotlib inline
```
```python
# Read yelp.csv into a DataFrame.
path = './data/yelp.csv'
yelp = pd.read_csv(path)

# Create a new DataFrame that only contains the 5-star and 1-star reviews.
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# Define X and y.
X = yelp_best_worst.text
y = yelp_best_worst.stars

# Split the new DataFrame into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```
```python
# The head of the original data
yelp.head()
```
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6oRAC4uyJCsJl1X0WZpVSA</td>
      <td>2012-06-14</td>
      <td>IESLBzqUCLdSzSqm0eCSxQ</td>
      <td>4</td>
      <td>love the gyro plate. Rice is so good and I als...</td>
      <td>review</td>
      <td>0hT2KtfLiobPvh6cDC8JQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>_1QQZuf4zZOyFCvXc0o6Vg</td>
      <td>2010-05-27</td>
      <td>G-WvGaISbqqaMHlNnByodA</td>
      <td>5</td>
      <td>Rosie, Dakota, and I LOVE Chaparral Dog Park!!...</td>
      <td>review</td>
      <td>uZetl9T0NcROGOyFfughhg</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6ozycU1RpktNG2-1BroVtw</td>
      <td>2012-01-05</td>
      <td>1uJFq2r5QfJG_6ExMRCaGw</td>
      <td>5</td>
      <td>General Manager Scott Petello is a good egg!!!...</td>
      <td>review</td>
      <td>vYmM4KTsC8ZfQBg-j5MWkw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

## Text Classification

As you proceed through this section, note that text classification is done in the same way as all other classification models. 
1. the text is vectorized into a set of numeric features. 
2. a standard machine learning classifier is applied
*NLP libraries often include vectorizers and ML models that work particularly well with text.* 

> We will refer to each piece of text we are trying to classify as a document.
> - For example, a document could refer to an email, book chapter, tweet, article, or text message.

**Text classification is the task of predicting which category or topic a text sample is from.**

We may want to identify:
- Is an article a sports or business story?
- Does an email have positive or negative sentiment?
- Is the rating of a recipe 1, 2, 3, 4, or 5 stars?

**Predictions are often made by using the words as features and the label as the target output.**

Starting out, we will make each unique word (across all documents) a single feature. In any given corpora, we may have hundreds of thousands of unique words, so we may have hundreds of thousands of features!

- For a given document, the numeric value of each feature could be the number of times the word appears in the document.
    - So, most features will have a value of zero, resulting in a sparse matrix of features.

- This technique for vectorizing text is referred to as a bag-of-words model. 
    - It is called bag of words because the document's structure is lost — as if the words are all jumbled up in a bag.
    - The first step to creating a bag-of-words model is to create a vocabulary of all possible words in the corpora.

> Alternatively, we could make each column an indicator column, which is 1 if the word is present in the document (no matter how many times) and 0 if not. This vectorization could be used to reduce the importance of repeated words. For example, a website search engine would be susceptible to spammers who load websites with repeated words. So, the search engine might use indicator columns as features rather than word counts.

**We need to consider several things to decide if bag-of-words is appropriate.**

- Does order of words matter?
- Does punctuation matter?
- Does upper or lower case matter?

## Demo: Text Processing in scikit-learn

- **Objective:** Demonstrate text classification.

<a id='count_vec'></a>


### Creating Features Using CountVectorizer

- **What:** Converts each document into a set of words and their counts.
- **Why:** To use a machine learning model, we must convert unstructured text into numeric features.
- **Notes:** Relatively easy with English language text, not as easy with some languages.


```python
# Use CountVectorizer to create document-term matrices from X_train and X_test.
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
```


```python
# Rows are documents, columns are terms (aka "tokens" or "features", individual words in this situation).
X_train_dtm.shape
```




    (3064, 16825)




```python
# Last 50 features
print(vect.get_feature_names()[-50:])
```

    [u'yyyyy', u'z11', u'za', u'zabba', u'zach', u'zam', u'zanella', u'zankou', u'zappos', u'zatsiki', u'zen', u'zero', u'zest', u'zexperience', u'zha', u'zhou', u'zia', u'zihuatenejo', u'zilch', u'zin', u'zinburger', u'zinburgergeist', u'zinc', u'zinfandel', u'zing', u'zip', u'zipcar', u'zipper', u'zippers', u'zipps', u'ziti', u'zoe', u'zombi', u'zombies', u'zone', u'zones', u'zoning', u'zoo', u'zoyo', u'zucca', u'zucchini', u'zuchinni', u'zumba', u'zupa', u'zuzu', u'zwiebel', u'zzed', u'\xe9clairs', u'\xe9cole', u'\xe9m']
    


```python
# Show vectorizer options.
vect
```




    CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
            dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)



[CountVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

One common method of reducing the number of features is converting all text to lowercase before generating features! Note that to a computer, `aPPle` is a different token/"word" than `apple`. So, by converting both to lowercase letters, it ensures fewer features will be generated. It might be useful not to convert them to lowercase if capitalization matters.


```python
# Don't convert to lowercase.
vect = CountVectorizer(lowercase=False)
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape
vect.get_feature_names()[-10:]
```




    [u'zoning',
     u'zoo',
     u'zucchini',
     u'zuchinni',
     u'zupa',
     u'zwiebel',
     u'zzed',
     u'\xc9cole',
     u'\xe9clairs',
     u'\xe9m']



<a id='countvectorizer-model'></a>


### Using CountVectorizer in a Model


```python
# Use default options for CountVectorizer.
vect = CountVectorizer()

# Create document-term matrices.
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Use Naive Bayes to predict the star rating.
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

# Calculate accuracy. 
# How accurately are we able to predict 1 vs 5 star reviews based on the review text?
print(metrics.accuracy_score(y_test, y_pred_class))
```

    0.918786692759
    


```python
y_test.value_counts()
```




    5    838
    1    184
    Name: stars, dtype: int64




```python
# Calculate null accuracy.
y_test_binary = np.where(y_test==5, 1, 0) # five stars become 1, one stars become 0
print 'Percent 5 Stars:', y_test_binary.mean()
print 'Percent 1 Stars:', 1 - y_test_binary.mean()
```

    Percent 5 Stars: 0.819960861057
    Percent 1 Stars: 0.180039138943
    

Our model predicted ~92% accuracy, which is an improvement over this baseline 82% accuracy (assuming our model always predicts 5 stars).

Let's look more into how the vectorizer works.


```python
# Notice how the data was transformed into this sparse matrix with 
# 1,022 datapoints and 16,825 features!
#   - Recall that vectorizations of text will be mostly zeros, since only a few unique words are in each document.
#   - For that reason, instead of storing all the zeros we only store non-zero values (inside the 'sparse matrix' data structure!).
#   - We have 3064 Yelp reviews in our training set.
#   - 16,825 unique words were found across all documents.

X_train_dtm
```




    <3064x16825 sparse matrix of type '<type 'numpy.int64'>'
    	with 237720 stored elements in Compressed Sparse Row format>




```python
# Let's take a look at the vocabulary that was generated, containing 16,825 unique words.
#   'vocabulary_' is a dictionary that converts each word to its index in the sparse matrix.
#   - For example, the word "four" is index #3230 in the sparse matrix.

vect.vocabulary_
```




    {u'mustachio': 9857,
     u'foul': 6081,
     u'four': 6089,
     u'bigee': 1706,
     u'woods': 16570,
     u'hanging': 6939,
     u'francesca': 6108,
     u'comically': 3246,
     u'increase': 7664,
     u'canes': 2456,
     u'shuro': 13415,
     u'originality': 10463,
     u'demoted': 4263,
     u'sinking': 13497,
     u'naturopathic': 9940,
     u'propane': 11679,
     u'eggrolls': 5029,
     u'tantalizing': 14789,
     u'leisurely': 8600,
     u'avondale': 1266,
     u'bringing': 2126,
     u'basics': 1476,
     u'wooden': 16568,
     u'wednesday': 16325,
     u'broiled': 2149,
     u'stereotypical': 14231,
     u'commented': 3254,
     u'specially': 13940,
     u'preface': 11471,
     u'272': 151,
     u'sustaining': 14624,
     u'scraped': 13031,
     u'lucious': 8928,
     u'errors': 5265,
     u'relieving': 12243,
     u'tiered': 15163,
     u'thunder': 15141,
     u'cooking': 3561,
     u'fossil': 6078,
     u'designing': 4328,
     u'marching': 9131,
     u'groupie': 6766,
     u'shocks': 13343,
     u'china': 2891,
     u'wagyu': 16175,
     u'affiliated': 550,
     u'chino': 2895,
     u'wiseguy': 16518,
     u'natured': 9938,
     u'kids': 8288,
     u'robata': 12589,
     u'controversy': 3533,
     u'advantages': 520,
     u'spotty': 14040,
     u'criticism': 3822,
     u'golden': 6554,
     u'replace': 12321,
     u'cocaine': 3166,
     u'slivers': 13616,
     u'dnf': 4630,
     u'browse': 2170,
     u'insecurity': 7783,
     u'favie': 5638,
     u'cannibal': 2461,
     u'coordinators': 3576,
     u'symphony': 14696,
     u'music': 9848,
     u'therefore': 15060,
     u'strike': 14352,
     u'spaying': 13925,
     u'paperwork': 10713,
     u'populations': 11352,
     u'hereby': 7150,
     u'yahoo': 16658,
     u'tastefully': 14827,
     u'intake': 7832,
     u'morally': 9731,
     u'locked': 8817,
     u'upbringings': 15836,
     u'locker': 8818,
     u'shelve': 13292,
     u'example': 5367,
     u'shirataki': 13323,
     u'omelets': 10343,
     u'wand': 16214,
     u'pints': 11146,
     u'yumness': 16763,
     u'kaprow': 8216,
     u'leotards': 8623,
     u'caution': 2631,
     u'want': 16220,
     u'cookers': 3557,
     u'chiccharon': 2851,
     u'pinto': 11145,
     u'absolute': 352,
     u'tqla': 15366,
     u'travel': 15428,
     u'subtleness': 14454,
     u'feature': 5662,
     u'mosear': 9755,
     u'swadee': 14631,
     u'terrill': 14986,
     u'goober': 6569,
     u'typed': 15606,
     u'minimized': 9530,
     u'adoptions': 505,
     u'wrong': 16639,
     u'hesistant': 7161,
     u'types': 15607,
     u'colorfully': 3217,
     u'sickening': 13425,
     u'baggage': 1339,
     u'18th': 77,
     u'timers': 15189,
     u'romper': 12634,
     u'623': 256,
     u'welcomed': 16353,
     u'benefited': 1636,
     u'rewarded': 12473,
     u'stabbed': 14099,
     u'yellowfin': 16685,
     u'welcomes': 16354,
     u'fit': 5837,
     u'screaming': 13040,
     u'fix': 5846,
     u'fig': 5744,
     u'wellness': 16357,
     u'fuddruckers': 6240,
     u'fin': 5781,
     u'easier': 4938,
     u'walet': 16193,
     u'heinritz': 7111,
     u'vouchers': 16150,
     u'glutten': 6525,
     u'effects': 5012,
     u'schools': 12991,
     u'sixteen': 13520,
     u'thyme': 15148,
     u'bartop': 1463,
     u'averse': 1257,
     u'ux10': 15908,
     u'volcano': 16132,
     u'payton': 10852,
     u'burial': 2273,
     u'directly': 4481,
     u'series': 13178,
     u'oohed': 10371,
     u'phoenicians': 11044,
     u'parasites': 10726,
     u'depositing': 4292,
     u'substantially': 14448,
     u'ring': 12529,
     u'drove': 4829,
     u'rr': 12696,
     u'oprah': 10401,
     u'mayan': 9271,
     u'rx': 12753,
     u'enviroment': 5234,
     u'mason': 9215,
     u'rd': 12010,
     u're': 12013,
     u'encourage': 5146,
     u'ra': 11887,
     u'rb': 12008,
     u'rc': 12009,
     u'rl': 12567,
     u'rm': 12568,
     u'foundation': 6084,
     u'threatened': 15114,
     u'tinkled': 15202,
     u'checked': 2799,
     u'estimate': 5294,
     u'chlorine': 2911,
     u'enormous': 5188,
     u'diabetic': 4384,
     u'shipped': 13321,
     u'disturbed': 4609,
     u'speedy': 13960,
     u'bannanas': 1413,
     u'speeds': 13959,
     u'purpose': 11796,
     u'brickwork': 2106,
     u'steroids': 14234,
     u'koch': 8360,
     u'channels': 2750,
     u'filibertos': 5761,
     u'kfc': 8268,
     u'infused': 7739,
     u'clarity': 3042,
     u'olds': 10332,
     u'basketball': 1481,
     u'renovated': 12296,
     u'service': 13193,
     u'zinfandel': 16798,
     u'reuben': 12447,
     u'needed': 9967,
     u'master': 9228,
     u'rewards': 12474,
     u'pimms': 11126,
     u'nirvana': 10075,
     u'mutilated': 9862,
     u'positively': 11384,
     u'trek': 15454,
     u'showed': 13391,
     u'basha': 1472,
     u'tree': 15451,
     u'bnd5erj': 1866,
     u'shower': 13392,
     u'exclaimed': 5391,
     u'friend': 6167,
     u'tres': 15463,
     u'feeling': 5679,
     u'runner': 12730,
     u'monkeys': 9697,
     u'spectrum': 13955,
     u'untrained': 15820,
     u'arousal': 1034,
     u'deamon': 4095,
     u'dozen': 4745,
     u'bleachers': 1800,
     u'uncouth': 15678,
     u'gripe': 6744,
     u'preservatives': 11518,
     u'recommended': 12100,
     u'amusing': 795,
     u'doors': 4701,
     u'brains': 2035,
     u'soldiered': 13775,
     u'menudo': 9397,
     u'committing': 3265,
     u'jammed': 8025,
     u'thrilled': 15123,
     u'vermicelli': 16020,
     u'prandial': 11446,
     u'wells': 16358,
     u'resonates': 12387,
     u'simplify': 13476,
     u'mouth': 9782,
     u'addict': 461,
     u'entry': 5228,
     u'03342': 8,
     u'regreted': 12194,
     u'camp': 2424,
     u'rotary': 12664,
     u'memberships': 9376,
     u'tech': 14888,
     u'prevention': 11551,
     u'scream': 13038,
     u'came': 2417,
     u'saying': 12940,
     u'condesa': 3385,
     u'meetings': 9349,
     u'teresa': 14970,
     u'x2': 16647,
     u'reschedule': 12358,
     u'indulges': 7703,
     u'participate': 10757,
     u'tempted': 14941,
     u'cheaply': 2792,
     u'rusticana': 12746,
     u'lessons': 8629,
     u'orleans': 10468,
     u'flowers': 5949,
     u'layout': 8532,
     u'quaint': 11827,
     u'luby': 8925,
     u'nordique': 10126,
     u'rico': 12500,
     u'bliss': 1820,
     u'rick': 12499,
     u'rich': 12491,
     u'rice': 12489,
     u'plate': 11217,
     u'dorrie': 4707,
     u'waaaaaay': 16162,
     u'honda': 7299,
     u'foremost': 6038,
     u'pocket': 11279,
     u'clumsiness': 3137,
     u'relish': 12245,
     u'increasing': 7665,
     u'jaguar': 8009,
     u'autistic': 1235,
     u'nicely': 10042,
     u'dipping': 4472,
     u'pretzel': 11546,
     u'patch': 10801,
     u'elote': 5077,
     u'boarded': 1872,
     u'heirloom': 7113,
     u'padi': 10624,
     u'pinot': 11142,
     u'pads': 10627,
     u'radius': 11912,
     u'48th': 220,
     u'pressured': 11529,
     u'lots': 8883,
     u'shittier': 13334,
     u'irs': 7955,
     u'rings': 12531,
     u'4hr': 224,
     u'targets': 14811,
     u'irk': 7938,
     u'yogurt': 16714,
     u'skimpy': 13552,
     u'rundown': 12729,
     u'splinter': 14008,
     u'nationwide': 9932,
     u'nature': 9937,
     u'smelled': 13658,
     u'merrels': 9408,
     u'lapping': 8466,
     u'smucker': 13687,
     u'chandler': 2741,
     u'defiance': 4181,
     u'tendons': 14958,
     u'debt': 4112,
     u'veer': 15970,
     u'disdain': 4530,
     u'country': 3667,
     u'planned': 11203,
     u'logic': 8826,
     u'argue': 1010,
     u'wrigley': 16625,
     u'asasda': 1086,
     u'hers': 7157,
     u'gleaming': 6501,
     u'pregnancy': 11482,
     u'blonde': 1833,
     u'priceline': 11558,
     u'grazing': 6689,
     u'underdone': 15689,
     u'union': 15755,
     u'fro': 6193,
     u'bothers': 1978,
     u'much': 9807,
     u'sommelier': 13812,
     u'gyoza': 6856,
     u'stadium': 14105,
     u'chugged': 2969,
     u'fry': 6223,
     u'tallest': 14763,
     u'dots': 4714,
     u'obese': 10236,
     u'recomend': 12096,
     u'spit': 14001,
     u'loraco': 8868,
     u'worker': 16581,
     u'dave': 4059,
     u'lahna': 8431,
     u'doubts': 4719,
     u'worked': 16580,
     u'regus': 12202,
     u'spin': 13990,
     u'wildcat': 16462,
     u'kennan': 8252,
     u'lalibela': 8438,
     u'administrative': 489,
     u'professionally': 11633,
     u'heidelberg': 7107,
     u'ridiculously': 12514,
     u'honeslty': 7303,
     u'saltiness': 12824,
     u'riparian': 12539,
     u'verges': 16016,
     u'conditioned': 3390,
     u'taunts': 14854,
     u'eighteen': 5037,
     u'upscale': 15861,
     u'hygienist': 7492,
     u'conditioner': 3391,
     u'hone': 7301,
     u'hong': 7310,
     u'portobello': 11373,
     u'split': 14009,
     u'european': 5319,
     u'supped': 14560,
     u'boiled': 1893,
     u'zoyo': 16813,
     u'jungle': 8180,
     u'qdoba': 11822,
     u'photographers': 11055,
     u'supper': 14561,
     u'acoustics': 425,
     u'corporate': 3610,
     u'massaging': 9221,
     u'plaque': 11211,
     u'booyah': 1945,
     u'capitol': 2491,
     u'appropriately': 974,
     u'sleepy': 13587,
     u'cannolis': 2463,
     u'snickered': 13715,
     u'subdued': 14431,
     u'lassi': 8489,
     u'rotates': 12667,
     u'previous': 11553,
     u'handshake': 6929,
     u'ham': 6901,
     u'had': 6868,
     u'hay': 7031,
     u'innocent': 7766,
     u'hap': 6951,
     u'has': 6999,
     u'hat': 7006,
     u'elevation': 5062,
     u'confection': 3401,
     u'calypso': 2414,
     u'shadow': 13227,
     u'desire': 4331,
     u'alice': 673,
     u'gangplank': 6330,
     u'pescatarian': 11003,
     u'festivities': 5707,
     u'vegtable': 15984,
     u'bulked': 2248,
     u'sebastian': 13091,
     u'attorney': 1204,
     u'creek': 3789,
     u'crowd': 3840,
     u'boyshorts': 2022,
     u'crown': 3845,
     u'begin': 1584,
     u'scoffed': 12997,
     u'enemies': 5165,
     u'perchance': 10947,
     u'bottom': 1985,
     u'yummie': 16754,
     u'unit': 15761,
     u'treadmill': 15442,
     u'creaminess': 3770,
     u'trite': 15487,
     u'completly': 3336,
     u'price': 11555,
     u'palatable': 10655,
     u'shaker': 13234,
     u'shakes': 13235,
     u'rep': 12308,
     u'substituted': 14450,
     u'defensive': 4178,
     u'farberware': 5601,
     u'losing': 8878,
     u'memorable': 9380,
     u'unhealthily': 15740,
     u'shaken': 13233,
     u'8yo': 301,
     u'benches': 1628,
     u'filipino': 5764,
     u'geranios': 6426,
     u'alber': 657,
     u'stoked': 14276,
     u'importantly': 7596,
     u'bath': 1491,
     u'raised': 11929,
     u'mochi': 9635,
     u'honeymoon': 7309,
     u'som': 13794,
     u'sol': 13771,
     u'soo': 13824,
     u'son': 13814,
     u'sop': 13834,
     u'magazines': 9010,
     u'marshals': 9187,
     u'wrap': 16616,
     u'sox': 13889,
     u'shoots': 13353,
     u'despised': 4340,
     u'fabric': 5523,
     u'waits': 16186,
     u'support': 14568,
     u'constantly': 3477,
     u'nova': 10172,
     u'tame': 14768,
     u'30p': 166,
     u'greatness': 6697,
     u'avocados': 1261,
     u'cerignola': 2689,
     u'yodels': 16710,
     u'diggity': 4437,
     u'magnificent': 9018,
     u'verde': 16011,
     u'paneled': 10691,
     u'azn': 1294,
     u'masquerading': 9216,
     u'extremity': 5504,
     u'inside': 7785,
     u'devices': 4369,
     u'paprika': 10717,
     u'nicest': 10044,
     u'servings': 13197,
     u'smashed': 13649,
     u'lolcat': 8833,
     u'passenger': 10777,
     u'disgrace': 4533,
     u'disclosing': 4505,
     u'calzones': 2416,
     u'sprouts': 14064,
     u'triangles': 15465,
     u'lees': 8582,
     u'whoreish': 16437,
     u'glacier': 6488,
     u'role': 12619,
     u'chocolatiers': 2919,
     u'entomologist': 5219,
     u'roll': 12620,
     u'intend': 7838,
     u'lollipops': 8835,
     u'palms': 10671,
     u'models': 9643,
     u'ramada': 11936,
     u'transported': 15419,
     u'scale': 12946,
     u'modelo': 9642,
     u'smelling': 13659,
     u'cleaver': 3075,
     u'loren': 8870,
     u'continentally': 3512,
     u'98': 310,
     u'pet': 11007,
     u'decision': 4135,
     u'steelhead': 14209,
     u'gown': 6618,
     u'bostic': 1970,
     u'childs': 2870,
     u'chain': 2708,
     u'whoever': 16428,
     u'hangovers': 6943,
     u'osp': 10477,
     u'bandito': 1399,
     u'yoohoo': 16723,
     u'burritos': 2286,
     u'skate': 13532,
     u'chair': 2712,
     u'muppet': 9832,
     u'midst': 9474,
     u'osf': 10476,
     u'ballet': 1375,
     u'crates': 3755,
     u'bicycles': 1699,
     u'burlington': 2274,
     u'jewelery': 8086,
     u'connecting': 3438,
     u'macho': 8987,
     u'snoooty': 13728,
     u'icing': 7512,
     u'jerk': 8072,
     u'entress': 5227,
     u'goldwell': 6556,
     u'choice': 2922,
     u'stays': 14191,
     u'exact': 5360,
     u'minute': 9545,
     u'cooks': 3563,
     u'epic': 5239,
     u'az88': 1292,
     u'minnie': 9537,
     u'skewed': 13540,
     u'leave': 8569,
     u'settle': 13204,
     u'skewer': 13541,
     u'loads': 8797,
     u'wannabes': 16219,
     u'spiritual': 13999,
     u'blade': 1776,
     u'bock': 1886,
     u'sigh': 13439,
     u'boca': 1883,
     u'sign': 13442,
     u'chopping': 2940,
     u'shirts': 13327,
     u'ogled': 10309,
     u'headset': 7051,
     u'chimichanga': 2886,
     u'burch': 2270,
     u'melt': 9367,
     u'wayward': 16303,
     u'baggins': 1341,
     u'lazily': 8534,
     u'crackling': 3731,
     u'boost': 1939,
     u'oldies': 10331,
     u'camel': 2418,
     u'continuing': 3517,
     u'understanding': 15698,
     u'upgrade': 15847,
     u'egypt': 5033,
     u'address': 474,
     u'dwindling': 4907,
     u'jura': 8186,
     u'passengers': 10778,
     u'breathing': 2082,
     u'redemption': 12121,
     u'brilliant': 2122,
     u'cusack': 3937,
     u'accomplished': 396,
     u'sorento': 13848,
     u'pibil': 11075,
     u'tasks': 14821,
     u'logical': 8827,
     u'texture': 15005,
     u'fake': 5566,
     u'umph': 15645,
     u'crammed': 3741,
     u'working': 16583,
     u'angry': 826,
     u'hyatt': 7483,
     u'papas': 10709,
     u'opposed': 10399,
     u'bender': 1630,
     u'citywide': 3024,
     u'scope': 13007,
     u'wicked': 16443,
     u'scratched': 13036,
     u'kreamery': 8377,
     u'cookouts': 3562,
     u'everywhere': 5344,
     u'scratcher': 13037,
     u'siebel': 13437,
     u'riders': 12508,
     u'originally': 10464,
     u'pretend': 11538,
     u'primer': 11576,
     u'following': 5996,
     u'zippers': 16803,
     u'mirrors': 9553,
     u'employess': 5120,
     u'stetson': 14235,
     u'awesome': 1280,
     u'parachute': 10719,
     u'matzo': 9260,
     u'allowed': 700,
     u'stole': 14277,
     u'listens': 8761,
     u'savoy': 12932,
     u'monitoring': 9693,
     u'grousing': 6771,
     u'savor': 12928,
     u'buttered': 2322,
     u'thanking': 15023,
     u'vancouver': 15944,
     u'cannon': 2464,
     u'teavana': 14887,
     u'umbria': 15641,
     u'fueled': 6244,
     u'matt': 9253,
     u'improving': 7618,
     u'revealed': 12453,
     u'brainless': 2034,
     u'sy': 14689,
     u'golfers': 6559,
     u'natural': 9935,
     u'conscious': 3449,
     u'consequently': 3452,
     u'sq': 14070,
     u'sp': 13892,
     u'sw': 14630,
     u'ordinarily': 10432,
     u'st': 14097,
     u'si': 13421,
     u'mango': 9100,
     u'so': 13738,
     u'sm': 13634,
     u'swollen': 14682,
     u'sb': 12943,
     u'sa': 12757,
     u'pulled': 11753,
     u'manga': 9098,
     u'se': 13060,
     u'sd': 13058,
     u'drunken': 4838,
     u'innocuous': 7767,
     u'drying': 4842,
     u'years': 16679,
     u'professors': 11635,
     u'frequents': 6149,
     u'firepit': 5819,
     u'tendency': 14950,
     u'tore': 15303,
     u'splurge': 14013,
     u'avid': 1259,
     u'jib': 8094,
     u'hillstone': 7205,
     u'toro': 15307,
     u'jim': 8098,
     u'tori': 15304,
     u'faves': 5637,
     u'tort': 15309,
     u'suspicion': 14620,
     u'stroll': 14372,
     u'constitutes': 3479,
     u'suspension': 14619,
     u'troubled': 15497,
     u'renaming': 12290,
     u'modestly': 9648,
     u'dpov': 4747,
     u'thereby': 15059,
     u'shacked': 13223,
     u'indigenous': 7686,
     u'overpowering': 10561,
     u'corroded': 3622,
     u'drilling': 4800,
     u'sorted': 13854,
     u'twilight': 15587,
     u'shouted': 13381,
     u'plunk': 11267,
     u'frank': 6116,
     u'didn': 4411,
     u'roasts': 12587,
     u'dispite': 4569,
     u'quarter': 11837,
     u'square': 14072,
     u'honduras': 7300,
     u'bursting': 2290,
     u'receipt': 12066,
     u'entering': 5201,
     u'beetle': 1574,
     u'krav': 8376,
     u'salads': 12789,
     u'overworked': 10583,
     u'grandad': 6644,
     u'cobs': 3163,
     u'seriously': 13180,
     u'trauma': 15427,
     u'cobb': 3161,
     u'internet': 7860,
     u'squares': 14073,
     u'mims': 9512,
     u'hairdresser': 6883,
     u'incentives': 7633,
     u'disrespect': 4579,
     u'lucrative': 8933,
     u'grandma': 6648,
     u'mimi': 9508,
     u'eh': 5035,
     u'patronage': 10820,
     u'workmanship': 16584,
     u'backside': 1317,
     u'telenovela': 14921,
     u'washingtons': 16264,
     u'downsides': 4741,
     u'gordon': 6595,
     u'flambeed': 5861,
     u'emotion': 5105,
     u'saving': 12926,
     u'ono': 10363,
     u'spoken': 14021,
     u'tolteca': 15259,
     u'one': 10354,
     u'looved': 8867,
     u'ony': 10368,
     u'mignon': 9483,
     u'open': 10379,
     u'city': 3023,
     u'wrath': 16622,
     u'bite': 1754,
     u'stuffed': 14403,
     u'definitly': 4190,
     u'spackling': 13899,
     u'bits': 1757,
     u'lingering': 8733,
     u'shawn': 13269,
     u'brewpub': 2097,
     u'almond': 704,
     u'fooled': 6017,
     u'iceberg': 7504,
     u'depressed': 4293,
     u'yelping': 16692,
     u'coats': 3158,
     u'trekking': 15455,
     u'rooste': 12647,
     u'individualized': 7690,
     u'wandering': 16217,
     u'damned': 4000,
     u'proactive': 11602,
     u'cider': 2986,
     u'addressing': 476,
     u'illness': 7554,
     u'sumptuous': 14521,
     u'turned': 15557,
     u'locations': 8815,
     u'jewels': 8088,
     u'mesquite': 9414,
     u'allen': 684,
     u'turner': 15558,
     u'50cents': 229,
     u'zoe': 16806,
     u'cigarettes': 2991,
     u'warriors': 16249,
     u'instructed': 7816,
     u'omfg': 10346,
     u'mazin': 9278,
     u'coincide': 3185,
     u'cidse': 2987,
     u'ironically': 7942,
     u'opposite': 10400,
     u'buffer': 2230,
     u'doudy': 4721,
     u'israeli': 7966,
     u'buffet': 2231,
     u'printed': 11584,
     u'lawn': 8520,
     u'average': 1253,
     u'phil': 11031,
     u'drive': 4809,
     u'jitters': 8101,
     u'wind': 16484,
     u'chiang': 2846,
     u'laws': 8522,
     u'lotus': 8885,
     u'2wice': 160,
     u'narcisstic': 9915,
     u'peaches': 10863,
     u'hopes': 7334,
     u'bright': 2119,
     u'inconsistent': 7654,
     u'aggressive': 584,
     u'calamari': 2389,
     u'ensembles': 5194,
     u'aimlessly': 613,
     u'newland': 10021,
     u'ouchh': 10489,
     u'rephrase': 12320,
     u'assistant': 1131,
     u'freezing': 6139,
     u'clang': 3036,
     u'simplistic': 13477,
     u'awaiting': 1268,
     u'resource': 12390,
     u'pimp': 11127,
     u'breaksfast': 2074,
     u'worried': 16593,
     u'tiki': 15174,
     u'pima': 11125,
     u'worries': 16594,
     u'tika': 15173,
     u'tortilla': 15314,
     u'vision': 16106,
     u'schmancy': 12983,
     u'artistically': 1071,
     u'anyways': 891,
     u'temporada': 14936,
     u'impressions': 7608,
     u'intoxicating': 7882,
     u'832': 290,
     u'planet': 11200,
     u'outclasses': 10498,
     u'mutton': 9864,
     u'taqueria': 14802,
     u'sites': 13512,
     u'vaccines': 15919,
     u'moldy': 9670,
     u'refreshed': 12162,
     u'vender': 15993,
     u'edamame': 4978,
     u'checkered': 2802,
     u'screen': 13042,
     u'custurd': 3951,
     u'awards': 1272,
     u'concentrated': 3364,
     u'busting': 2307,
     u'many': 9120,
     u'45min': 214,
     u'plants': 11210,
     u'kahlo': 8203,
     u'unflattering': 15730,
     u'mani': 9103,
     u'grooming': 6759,
     u'potstickers': 11413,
     u'allowance': 699,
     u'yearly': 16676,
     u'loveliest': 8902,
     u'compels': 3306,
     u'betweeen': 1670,
     u'zihuatenejo': 16792,
     u'3000': 164,
     u'cokes': 3188,
     u'considers': 3459,
     u'caring': 2534,
     u'west': 16365,
     u'artichokes': 1062,
     u'camelback': 2419,
     u'reflex': 12161,
     u'wants': 16225,
     u'mekong': 9358,
     u'centerpiece': 2677,
     u'photos': 11060,
     u'former': 6061,
     u'32': 172,
     u'consultant': 3486,
     u'want_': 16221,
     u'endures': 5164,
     u'policies': 11302,
     u'pollo': 11313,
     u'newspaper': 10028,
     u'situation': 13517,
     u'ive': 7986,
     u'pinko': 11139,
     u'checkeed': 2800,
     u'purse': 11800,
     u'moreso': 9738,
     u'technology': 14896,
     u'ivy': 7988,
     u'debilitating': 4108,
     u'ingrained': 7744,
     u'wiring': 16514,
     u'everday': 5334,
     u'homophobia': 7296,
     u'wrappers': 16619,
     u'visually': 16116,
     u'hideaway': 7177,
     u'singapore': 13489,
     u'edges': 4984,
     u'barber': 1430,
     u'advertisement': 528,
     u'six': 13519,
     u'recapture': 12063,
     u'booster': 1940,
     u'mohawk': 9654,
     u'awesomeness': 1282,
     u'customized': 3949,
     u'veal': 15969,
     u'loosens': 8865,
     u'norterra': 10135,
     u'customizes': 3950,
     u'steamed': 14204,
     u'being': 1599,
     u'rest': 12407,
     u'interrupt': 7864,
     u'forewarned': 6042,
     u'ngoc': 10034,
     u'cilantro': 2994,
     u'mesha': 9412,
     u'starving': 14168,
     u'crusty': 3872,
     u'around': 1033,
     u'sums': 14522,
     u'crusts': 3871,
     u'dart': 4045,
     u'dark': 4035,
     u'regio': 12185,
     u'traffic': 15382,
     u'preference': 11476,
     u'vacuum': 15920,
     u'world': 16588,
     u'dara': 4027,
     u'cesspools': 2697,
     u'pizzaria': 11175,
     u'sensational': 13155,
     u'intel': 7836,
     u'crostini': 3837,
     u'exploit': 5469,
     u'rigmarole': 12527,
     u'gimme': 6457,
     u'rachael': 11897,
     u'kennel': 8254,
     u'accidentally': 381,
     u'seating': 13086,
     u'learning': 8558,
     u'zupa': 16818,
     u'pickled': 11083,
     u'tvs': 15571,
     u'lobster': 8804,
     u'demeaning': 4252,
     u'pickles': 11084,
     u'divine': 4621,
     u'cavity': 2641,
     u'memories': 9382,
     u'sucha': 14472,
     u'911': 305,
     u'refer': 12141,
     u'biased': 1694,
     u'transformers': 15405,
     u'zest': 16787,
     u'internship': 7861,
     u'corkage': 3598,
     u'throwback': 15132,
     u'thailand': 15014,
     u'prepared': 11498,
     u'arroz': 1055,
     u'origins': 10466,
     u'package': 10612,
     u'champagnes': 2729,
     u'nastily': 9919,
     u'perturbed': 10998,
     u'lively': 8777,
     u'homeless': 7285,
     u'apologize': 905,
     u'hatha': 7012,
     u'goliath': 6561,
     u'bubbly': 2201,
     u'jerks': 8073,
     u'hep': 7141,
     u'her': 7142,
     u'hes': 7160,
     u'aguas': 596,
     u'hey': 7166,
     u'lounging': 8896,
     u'hee': 7095,
     u'sealed': 13066,
     u'brazilian': 2059,
     ...}




```python
# Finally, let's convert the sparse matrix to a typical ndarray using .toarray()
#   - Remember, this takes up a lot more memory than the sparse matrix! However, this conversion is sometimes necessary.

X_test_dtm.toarray()
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ..., 
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python
# We will use this function below for simplicity.

# Define a function that accepts a vectorizer and calculates the accuracy.
def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print('Features: ', X_train_dtm.shape[1])
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
```


```python
# min_df ignores words that occur less than twice ('df' means "document frequency").
vect = CountVectorizer(min_df=2, max_features=10000)
tokenize_test(vect)
```

    ('Features: ', 8783)
    ('Accuracy: ', 0.92465753424657537)
    

Let's take a look next at other ways of preprocessing text!

- **Objective:** Demonstrate common text preprocessing techniques.

<a id='ngrams'></a>
### N-Grams

N-grams are features which consist of N consecutive words. This is useful because using the bag-of-words model, treating `data scientist` as a single feature has more meaning than having two independent features `data` and `scientist`!

Example:
```
my cat is awesome
Unigrams (1-grams): 'my', 'cat', 'is', 'awesome'
Bigrams (2-grams): 'my cat', 'cat is', 'is awesome'
Trigrams (3-grams): 'my cat is', 'cat is awesome'
4-grams: 'my cat is awesome'
```

- **ngram_range:** tuple (min_n, max_n)
- The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.


```python
# Include 1-grams and 2-grams.
vect = CountVectorizer(ngram_range=(1, 2))
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape
```




    (3064, 169847)



We can start to see how supplementing our features with n-grams can lead to more feature columns. When we produce n-grams from a document with $W$ words, we add an additional $(n-W+1)$ features (at most). That said, be careful — when we compute n-grams from an entire corpus, the number of _unique_ n-grams could be vastly higher than the number of _unique_ unigrams! This could cause an undesired feature explosion.

Although we sometimes add important new features that have meaning such as `data scientist`, many of the new features will just be noise. So, particularly if we do not have much data, adding n-grams can actually decrease model performance. This is because if each n-gram is only present once or twice in the training set, we are effectively adding mostly noisy features to the mix.


```python
# Last 50 features
print(vect.get_feature_names()[-50:])
```

    [u'zone out', u'zone when', u'zones', u'zones dolls', u'zoning', u'zoning issues', u'zoo', u'zoo and', u'zoo is', u'zoo not', u'zoo the', u'zoo ve', u'zoyo', u'zoyo for', u'zucca', u'zucca appetizer', u'zucchini', u'zucchini and', u'zucchini bread', u'zucchini broccoli', u'zucchini carrots', u'zucchini fries', u'zucchini pieces', u'zucchini strips', u'zucchini veal', u'zucchini very', u'zucchini with', u'zuchinni', u'zuchinni again', u'zuchinni the', u'zumba', u'zumba class', u'zumba or', u'zumba yogalates', u'zupa', u'zupa flavors', u'zuzu', u'zuzu in', u'zuzu is', u'zuzu the', u'zwiebel', u'zwiebel kr\xe4uter', u'zzed', u'zzed in', u'\xe9clairs', u'\xe9clairs napoleons', u'\xe9cole', u'\xe9cole len\xf4tre', u'\xe9m', u'\xe9m all']
    

<a id='stopwords'></a>

### Stop-Word Removal

- **What:** This process is used to remove common words that will likely appear in any text.
- **Why:** Because common words exist in most documents, they likely only add noise to your model and should be removed.

**What are stop words?**
Stop words are some of the most common words in a language. They are used so that a sentence makes sense grammatically, such as prepositions and determiners, e.g., "to," "the," "and." However, they are so commonly used that they are generally worthless for predicting the class of a document. Since "a" appears in spam and non-spam emails, for example, it would only contribute noise to our model.

Example: 

> 1. Original sentence: "The dog jumped over the fence"  
> 2. After stop-word removal: "dog jumped over fence"

The fact that there is a fence and a dog jumped over it can be derived with or without stop words.


```python
# Show vectorizer options.
vect
```




    CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
            dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 2), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)



- **stop_words:** string {`english`}, list, or None (default)
- If `english`, a built-in stop word list for English is used.
- If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
- If None, no stop words will be used. `max_df` can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms. (If `max_df` = 0.7, then if > 70% of documents contain a word it will not be included in the feature set!)


```python
# Remove English stop words.
vect = CountVectorizer(stop_words='english')
tokenize_test(vect)
vect.get_params()
```

    ('Features: ', 16528)
    ('Accuracy: ', 0.91585127201565553)
    




    {'analyzer': u'word',
     'binary': False,
     'decode_error': u'strict',
     'dtype': numpy.int64,
     'encoding': u'utf-8',
     'input': u'content',
     'lowercase': True,
     'max_df': 1.0,
     'max_features': None,
     'min_df': 1,
     'ngram_range': (1, 1),
     'preprocessor': None,
     'stop_words': 'english',
     'strip_accents': None,
     'token_pattern': u'(?u)\\b\\w\\w+\\b',
     'tokenizer': None,
     'vocabulary': None}




```python
# Set of stop words
print(vect.get_stop_words())
```

    frozenset(['all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'fifty', 'four', 'not', 'own', 'through', 'yourselves', 'go', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere', 'with', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'under', 'ours', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very', 'de', 'none', 'cannot', 'every', 'whether', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'several', 'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone', 'either', 'each', 'become', 'thereupon', 'sometime', 'side', 'two', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up', 'namely', 'towards', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its', 'everything', 'behind', 'un', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she', 'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere', 'although', 'found', 'alone', 're', 'along', 'fifteen', 'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others', 'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover', 'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due', 'been', 'next', 'anyone', 'eleven', 'much', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty', 'these', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'toward', 'my', 'something', 'and', 'whereafter', 'whenever', 'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps', 'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon', 'eight', 'but', 'serious', 'nothing', 'such', 'your', 'why', 'a', 'off', 'whereby', 'third', 'i', 'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'whereas', 'once'])
    

<a id='cvec_opt'></a>
### Other CountVectorizer Options

- `max_features`: int or None, default=None
- If not None, build a vocabulary that only consider the top `max_features` ordered by term frequency across the corpus. This allows us to keep more common n-grams and remove ones that may appear once. If we include words that only occur once, this can lead to said features being highly associated with a class and cause overfitting.


```python
# Remove English stop words and only keep 100 features.
vect = CountVectorizer(stop_words='english', max_features=100)
tokenize_test(vect)
```

    ('Features: ', 100)
    ('Accuracy: ', 0.86986301369863017)
    


```python
# All 100 features
print(vect.get_feature_names())
```

    [u'amazing', u'area', u'atmosphere', u'awesome', u'bad', u'bar', u'best', u'better', u'big', u'came', u'cheese', u'chicken', u'clean', u'coffee', u'come', u'day', u'definitely', u'delicious', u'did', u'didn', u'dinner', u'don', u'eat', u'excellent', u'experience', u'favorite', u'feel', u'food', u'free', u'fresh', u'friendly', u'friends', u'going', u'good', u'got', u'great', u'happy', u'home', u'hot', u'hour', u'just', u'know', u'like', u'little', u'll', u'location', u'long', u'looking', u'lot', u'love', u'lunch', u'make', u'meal', u'menu', u'minutes', u'need', u'new', u'nice', u'night', u'order', u'ordered', u'people', u'perfect', u'phoenix', u'pizza', u'place', u'pretty', u'prices', u'really', u'recommend', u'restaurant', u'right', u'said', u'salad', u'sandwich', u'sauce', u'say', u'service', u'staff', u'store', u'sure', u'table', u'thing', u'things', u'think', u'time', u'times', u'took', u'town', u'tried', u'try', u've', u'wait', u'want', u'way', u'went', u'wine', u'work', u'worth', u'years']
    

Just like with all other models, more features does not mean a better model. So, we must tune our feature generator to remove features whose predictive capability is none or very low.

In this case, there is roughly a 1.6% increase in accuracy when we double the n-gram size and increase our max features by 1,000-fold. Note that if we restrict it to only unigrams, then the accuracy increases even more! So, bigrams were very likely adding more noise than signal. 

In the end, by only using 16,000 unigram features we came away with a much smaller, simpler, and easier-to-think-about model which also resulted in higher accuracy.


```python
# Include 1-grams and 2-grams, and limit the number of features.

print '1-grams and 2-grams, up to 100K features:'
vect = CountVectorizer(ngram_range=(1, 2), max_features=100000)
tokenize_test(vect)

print
print '1-grams only, up to 100K features:'
vect = CountVectorizer(ngram_range=(1, 1), max_features=100000)
tokenize_test(vect)
```

    1-grams and 2-grams, up to 100K features:
    ('Features: ', 100000)
    ('Accuracy: ', 0.88551859099804309)
    
    1-grams only, up to 100K features:
    ('Features: ', 16825)
    ('Accuracy: ', 0.91878669275929548)
    

- `min_df`: Float in range [0.0, 1.0] or int, default=1
- When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts.


```python
# Include 1-grams and 2-grams, and only include terms that appear at least two times.
vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
tokenize_test(vect)
```

    ('Features: ', 43957)
    ('Accuracy: ', 0.93248532289628183)
    

<a id='textblob'></a>
## Introduction to TextBlob

You should already have downloaded TextBlob, a Python library used to explore common NLP tasks. If you haven’t, please return to [this step](#textblob_install) for instructions on how to do so. We’ll be using this to organize our corpora for analysis.

As mentioned earlier, you can read more on the [TextBlob website](https://textblob.readthedocs.io/en/dev/).


```python
# Print the first review.
print(yelp_best_worst.text[0])
```

    My wife took me here on my birthday for breakfast and it was excellent.  The weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.  Our waitress was excellent and our food arrived quickly on the semi-busy Saturday morning.  It looked like the place fills up pretty quickly so the earlier you get here the better.
    
    Do yourself a favor and get their Bloody Mary.  It was phenomenal and simply the best I've ever had.  I'm pretty sure they only use ingredients from their garden and blend them fresh when you order it.  It was amazing.
    
    While EVERYTHING on the menu looks excellent, I had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious.  It came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete.  It was the best "toast" I've ever had.
    
    Anyway, I can't wait to go back!
    


```python
# Save it as a TextBlob object.
review = TextBlob(yelp_best_worst.text[0])
```


```python
# List the words.
review.words
```




    WordList(['My', 'wife', 'took', 'me', 'here', 'on', 'my', 'birthday', 'for', 'breakfast', 'and', 'it', 'was', 'excellent', 'The', 'weather', 'was', 'perfect', 'which', 'made', 'sitting', 'outside', 'overlooking', 'their', 'grounds', 'an', 'absolute', 'pleasure', 'Our', 'waitress', 'was', 'excellent', 'and', 'our', 'food', 'arrived', 'quickly', 'on', 'the', 'semi-busy', 'Saturday', 'morning', 'It', 'looked', 'like', 'the', 'place', 'fills', 'up', 'pretty', 'quickly', 'so', 'the', 'earlier', 'you', 'get', 'here', 'the', 'better', 'Do', 'yourself', 'a', 'favor', 'and', 'get', 'their', 'Bloody', 'Mary', 'It', 'was', 'phenomenal', 'and', 'simply', 'the', 'best', 'I', "'ve", 'ever', 'had', 'I', "'m", 'pretty', 'sure', 'they', 'only', 'use', 'ingredients', 'from', 'their', 'garden', 'and', 'blend', 'them', 'fresh', 'when', 'you', 'order', 'it', 'It', 'was', 'amazing', 'While', 'EVERYTHING', 'on', 'the', 'menu', 'looks', 'excellent', 'I', 'had', 'the', 'white', 'truffle', 'scrambled', 'eggs', 'vegetable', 'skillet', 'and', 'it', 'was', 'tasty', 'and', 'delicious', 'It', 'came', 'with', '2', 'pieces', 'of', 'their', 'griddled', 'bread', 'with', 'was', 'amazing', 'and', 'it', 'absolutely', 'made', 'the', 'meal', 'complete', 'It', 'was', 'the', 'best', 'toast', 'I', "'ve", 'ever', 'had', 'Anyway', 'I', 'ca', "n't", 'wait', 'to', 'go', 'back'])




```python
# List the sentences.
review.sentences
```




    [Sentence("My wife took me here on my birthday for breakfast and it was excellent."),
     Sentence("The weather was perfect which made sitting outside overlooking their grounds an absolute pleasure."),
     Sentence("Our waitress was excellent and our food arrived quickly on the semi-busy Saturday morning."),
     Sentence("It looked like the place fills up pretty quickly so the earlier you get here the better."),
     Sentence("Do yourself a favor and get their Bloody Mary."),
     Sentence("It was phenomenal and simply the best I've ever had."),
     Sentence("I'm pretty sure they only use ingredients from their garden and blend them fresh when you order it."),
     Sentence("It was amazing."),
     Sentence("While EVERYTHING on the menu looks excellent, I had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious."),
     Sentence("It came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete."),
     Sentence("It was the best "toast" I've ever had."),
     Sentence("Anyway, I can't wait to go back!")]




```python
# Some string methods are available.
review.lower()
```




    TextBlob("my wife took me here on my birthday for breakfast and it was excellent.  the weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.  our waitress was excellent and our food arrived quickly on the semi-busy saturday morning.  it looked like the place fills up pretty quickly so the earlier you get here the better.
    
    do yourself a favor and get their bloody mary.  it was phenomenal and simply the best i've ever had.  i'm pretty sure they only use ingredients from their garden and blend them fresh when you order it.  it was amazing.
    
    while everything on the menu looks excellent, i had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious.  it came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete.  it was the best "toast" i've ever had.
    
    anyway, i can't wait to go back!")



<a id='stem'></a>
## Stemming and Lemmatization

Stemming is a crude process of removing common endings from sentences, such as "s", "es", "ly", "ing", and "ed".

- **What:** Reduce a word to its base/stem/root form.
- **Why:** This intelligently reduces the number of features by grouping together (hopefully) related words.
- **Notes:**
    - Stemming uses a simple and fast rule-based approach.
    - Stemmed words are usually not shown to users (used for analysis/indexing).
    - Some search engines treat words with the same stem as synonyms.


```python
# Initialize stemmer.
stemmer = SnowballStemmer('english')

# Stem each word.
print([stemmer.stem(word) for word in review.words])
```

    [u'my', u'wife', u'took', u'me', u'here', u'on', u'my', u'birthday', u'for', u'breakfast', u'and', u'it', u'was', u'excel', u'the', u'weather', u'was', u'perfect', u'which', u'made', u'sit', u'outsid', u'overlook', u'their', u'ground', u'an', u'absolut', u'pleasur', u'our', u'waitress', u'was', u'excel', u'and', u'our', u'food', u'arriv', u'quick', u'on', u'the', u'semi-busi', u'saturday', u'morn', u'it', u'look', u'like', u'the', u'place', u'fill', u'up', u'pretti', u'quick', u'so', u'the', u'earlier', u'you', u'get', u'here', u'the', u'better', u'do', u'yourself', u'a', u'favor', u'and', u'get', u'their', u'bloodi', u'mari', u'it', u'was', u'phenomen', u'and', u'simpli', u'the', u'best', u'i', u've', u'ever', u'had', u'i', u"'m", u'pretti', u'sure', u'they', u'onli', u'use', u'ingredi', u'from', u'their', u'garden', u'and', u'blend', u'them', u'fresh', u'when', u'you', u'order', u'it', u'it', u'was', u'amaz', u'while', u'everyth', u'on', u'the', u'menu', u'look', u'excel', u'i', u'had', u'the', u'white', u'truffl', u'scrambl', u'egg', u'veget', u'skillet', u'and', u'it', u'was', u'tasti', u'and', u'delici', u'it', u'came', u'with', u'2', u'piec', u'of', u'their', u'griddl', u'bread', u'with', u'was', u'amaz', u'and', u'it', u'absolut', u'made', u'the', u'meal', u'complet', u'it', u'was', u'the', u'best', u'toast', u'i', u've', u'ever', u'had', u'anyway', u'i', u'ca', u"n't", u'wait', u'to', u'go', u'back']
    

Some examples you can see are "excellent" stemmed to "excel" and "amazing" stemmed to "amaz".

Lemmatization is a more refined process that uses specific language and grammar rules to derive the root of a word.  

This is useful for words that do not share an obvious root such as "better" and "best".

- **What:** Lemmatization derives the canonical form ("lemma") of a word.
- **Why:** It can be better than stemming.
- **Notes:** Uses a dictionary-based approach (slower than stemming).


```python
# Assume every word is a noun.
print([word.lemmatize() for word in review.words])
```

    ['My', 'wife', 'took', 'me', 'here', 'on', 'my', 'birthday', 'for', 'breakfast', 'and', 'it', u'wa', 'excellent', 'The', 'weather', u'wa', 'perfect', 'which', 'made', 'sitting', 'outside', 'overlooking', 'their', u'ground', 'an', 'absolute', 'pleasure', 'Our', 'waitress', u'wa', 'excellent', 'and', 'our', 'food', 'arrived', 'quickly', 'on', 'the', 'semi-busy', 'Saturday', 'morning', 'It', 'looked', 'like', 'the', 'place', u'fill', 'up', 'pretty', 'quickly', 'so', 'the', 'earlier', 'you', 'get', 'here', 'the', 'better', 'Do', 'yourself', 'a', 'favor', 'and', 'get', 'their', 'Bloody', 'Mary', 'It', u'wa', 'phenomenal', 'and', 'simply', 'the', 'best', 'I', "'ve", 'ever', 'had', 'I', "'m", 'pretty', 'sure', 'they', 'only', 'use', u'ingredient', 'from', 'their', 'garden', 'and', 'blend', 'them', 'fresh', 'when', 'you', 'order', 'it', 'It', u'wa', 'amazing', 'While', 'EVERYTHING', 'on', 'the', 'menu', u'look', 'excellent', 'I', 'had', 'the', 'white', 'truffle', 'scrambled', u'egg', 'vegetable', 'skillet', 'and', 'it', u'wa', 'tasty', 'and', 'delicious', 'It', 'came', 'with', '2', u'piece', 'of', 'their', 'griddled', 'bread', 'with', u'wa', 'amazing', 'and', 'it', 'absolutely', 'made', 'the', 'meal', 'complete', 'It', u'wa', 'the', 'best', 'toast', 'I', "'ve", 'ever', 'had', 'Anyway', 'I', 'ca', "n't", 'wait', 'to', 'go', 'back']
    

Some examples you can see are "filled" lemmatized to "fill" and "was" lemmatized to "wa".



```python
# Assume every word is a verb.
print([word.lemmatize(pos='v') for word in review.words])
```

    ['My', 'wife', u'take', 'me', 'here', 'on', 'my', 'birthday', 'for', 'breakfast', 'and', 'it', u'be', 'excellent', 'The', 'weather', u'be', 'perfect', 'which', u'make', u'sit', 'outside', u'overlook', 'their', u'ground', 'an', 'absolute', 'pleasure', 'Our', 'waitress', u'be', 'excellent', 'and', 'our', 'food', u'arrive', 'quickly', 'on', 'the', 'semi-busy', 'Saturday', 'morning', 'It', u'look', 'like', 'the', 'place', u'fill', 'up', 'pretty', 'quickly', 'so', 'the', 'earlier', 'you', 'get', 'here', 'the', 'better', 'Do', 'yourself', 'a', 'favor', 'and', 'get', 'their', 'Bloody', 'Mary', 'It', u'be', 'phenomenal', 'and', 'simply', 'the', 'best', 'I', "'ve", 'ever', u'have', 'I', "'m", 'pretty', 'sure', 'they', 'only', 'use', 'ingredients', 'from', 'their', 'garden', 'and', 'blend', 'them', 'fresh', 'when', 'you', 'order', 'it', 'It', u'be', u'amaze', 'While', 'EVERYTHING', 'on', 'the', 'menu', u'look', 'excellent', 'I', u'have', 'the', 'white', 'truffle', u'scramble', u'egg', 'vegetable', 'skillet', 'and', 'it', u'be', 'tasty', 'and', 'delicious', 'It', u'come', 'with', '2', u'piece', 'of', 'their', u'griddle', 'bread', 'with', u'be', u'amaze', 'and', 'it', 'absolutely', u'make', 'the', 'meal', 'complete', 'It', u'be', 'the', 'best', 'toast', 'I', "'ve", 'ever', u'have', 'Anyway', 'I', 'ca', "n't", 'wait', 'to', 'go', 'back']
    

Some examples you can see are "was" lemmatized to "be" and "arrived" lemmatized to "arrive".

**More Lemmatization and Stemming Examples**

|Lemmatization|Stemming|
|-------------|---------|
|shouted → shout|badly → bad|
|best → good|computing → comput|
|better → good|computed → comput|
|good → good|wipes → wip|
|wiping → wipe|wiped → wip|
|hidden → hide|wiping → wip|

### Activity: Knowledge Check
- What other words or phrases might cause problems with stemming? Why?
- What other words or phrases might cause problems with lemmatization? Why?

----




```python
# Define a function that accepts text and returns a list of lemmas.
def split_into_lemmas(text):
    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]
```


```python
# Use split_into_lemmas as the feature extraction function (Warning: SLOW!).
vect = CountVectorizer(analyzer=split_into_lemmas, decode_error='replace')
tokenize_test(vect)
```

    ('Features: ', 16452)
    ('Accuracy: ', 0.92074363992172215)
    


```python
# Last 50 features
print(vect.get_feature_names()[-50:])
```

    [u'yuyuyummy', u'yuzu', u'z', u'z-grill', u'z11', u'zach', u'zam', u'zanella', u'zankou', u'zappos', u'zatsiki', u'zen', u'zen-like', u'zero', u'zero-star', u'zest', u'zexperience', u'zha', u'zhou', u'zia', u'zilch', u'zin', u'zinburger', u'zinburgergeist', u'zinc', u'zinfandel', u'zing', u'zip', u'zipcar', u'zipper', u'zipps', u'ziti', u'zoe', u'zombi', u'zombie', u'zone', u'zoning', u'zoo', u'zoyo', u'zucca', u'zucchini', u'zuchinni', u'zumba', u'zupa', u'zuzu', u'zwiebel-kr\xe4uter', u'zzed', u'\xe9clairs', u'\xe9cole', u'\xe9m']
    

With all the available options for `CountVectorizer()`, you may wonder how to decide which to use! It's true that you can sometimes reason about which preprocessing techniques might work best. However, you will often not know for sure without trying out many different combinations and comparing their accuracies. 

> Keep in mind that you should constantly be thinking about the result of each preprocessing step instead of blindly trying them without thinking. Does each type of preprocessing "makes sense" with the input data you are using? Is it likely to keep intact the signal and remove noise?

<a id='tfidf'></a>
## Term Frequency–Inverse Document Frequency (TF–IDF)

While a Count Vectorizer simply totals up the number of times a "word" appears in a document, the more complex TF-IDF Vectorizer analyzes the uniqueness of words between documents to find distinguishing characteristics. 
     
- **What:** Term frequency–inverse document frequency (TF–IDF) computes the "relative frequency" with which a word appears in a document, compared to its frequency across all documents.
- **Why:** It's more useful than "term frequency" for identifying "important" words in each document (high frequency in that document, low frequency in other documents).
- **Notes:** It's used for search-engine scoring, text summarization, and document clustering.


```python
# Example documents
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']
```


```python
# Term frequency
vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
tf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cab</th>
      <th>call</th>
      <th>me</th>
      <th>please</th>
      <th>tonight</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Document frequency
vect = CountVectorizer(binary=True)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cab</th>
      <th>call</th>
      <th>me</th>
      <th>please</th>
      <th>tonight</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Term frequency–inverse document frequency (simple version)
tf/df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cab</th>
      <th>call</th>
      <th>me</th>
      <th>please</th>
      <th>tonight</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The higher the TF–IDF value, the more "important" the word is to that specific document. Here, "cab" is the most important and unique word in document 1, while "please" is the most important and unique word in document 2. TF–IDF is often used for training as a replacement for word count.


```python
# TfidfVectorizer
vect = TfidfVectorizer()
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cab</th>
      <th>call</th>
      <th>me</th>
      <th>please</th>
      <th>tonight</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.385372</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.652491</td>
      <td>0.652491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.720333</td>
      <td>0.425441</td>
      <td>0.547832</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.266075</td>
      <td>0.342620</td>
      <td>0.901008</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



**More details:** [TF–IDF is about what matters](http://planspace.org/20150524-tfidf_is_about_what_matters/)

<a id='yelp_tfidf'></a>
## Using TF–IDF to Summarize a Yelp Review

Reddit's autotldr uses the [SMMRY](http://smmry.com/about) algorithm, which is based on TF–IDF.


```python
# Create a document-term matrix using TF–IDF.
vect = TfidfVectorizer(stop_words='english')

# Fit transform Yelp data.
dtm = vect.fit_transform(yelp.text)
features = vect.get_feature_names()
dtm.shape
```




    (10000, 28880)




```python
def summarize():
    
    # Choose a random review that is at least 300 characters.
    review_length = 0
    while review_length < 300:
        review_id = np.random.randint(0, len(yelp))
        review_text = yelp.text[review_id]
        #review_text = unicode(yelp.text[review_id], 'utf-8')
        review_length = len(review_text)
    
    # Create a dictionary of words and their TF–IDF scores.
    word_scores = {}
    for word in TextBlob(review_text).words:
        word = word.lower()
        if word in features:
            word_scores[word] = dtm[review_id, features.index(word)]
    
    # Print words with the top five TF–IDF scores.
    print('TOP SCORING WORDS:')
    top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, score in top_scores:
        print(word)
    
    # Print five random words.
    print('\n' + 'RANDOM WORDS:')
    random_words = np.random.choice(list(word_scores.keys()), size=5, replace=False)
    for word in random_words:
        print(word)
    
    # Print the review.
    print('\n' + review_text)
```


```python
summarize()
```

    TOP SCORING WORDS:
    times
    drunkard
    dammit
    gringos
    friends
    
    RANDOM WORDS:
    dammit
    professional
    suggest
    drinker
    drunkard
    
    Dammit - I have been here a dozen times and it's been so "mixed"... The best of times, and the worst of times... though the last two times (for late night drinking with friends) have been pretty fun... Drinks are cheap and the crowd is "lively".
    
    So, for now, I will suggest Dos Gringos to the college drunkard and/or professional drinker after 9pm. Bring some friends!
    

<a id='sentiment'></a>
## Sentiment Analysis

Understanding how positive or negative a review is. There are many ways in practice to compute a sentiment value. For example:

- Have a list of "positive" words and a list of "negative" words and count how many occur in a document. 
- Train a classifier given many examples of "positive" documents and "negative" documents. 
    - Note that this technique is often just an automated way to derive the first (e.g., using bag-of-words with logistic regression, a coefficient is assigned to each word!).

For the most accurate sentiment analysis, you will want to train a custom sentiment model based on documents that are particular to your application. Generic models (such as the one we are about to use!) often do not work as well as hoped.

As we will do below, always make sure you double-check that the algorithm is working by manually verifying that scores correctly correspond to positive/negative reviews! Otherwise, you may be using numbers that are not accurate.


```python
print(review)
```

    My wife took me here on my birthday for breakfast and it was excellent.  The weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.  Our waitress was excellent and our food arrived quickly on the semi-busy Saturday morning.  It looked like the place fills up pretty quickly so the earlier you get here the better.
    
    Do yourself a favor and get their Bloody Mary.  It was phenomenal and simply the best I've ever had.  I'm pretty sure they only use ingredients from their garden and blend them fresh when you order it.  It was amazing.
    
    While EVERYTHING on the menu looks excellent, I had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious.  It came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete.  It was the best "toast" I've ever had.
    
    Anyway, I can't wait to go back!
    


```python
# Polarity ranges from -1 (most negative) to 1 (most positive).
review.sentiment.polarity
```




    0.40246913580246907




```python
# Understanding the apply method
yelp['length'] = yelp.text.apply(len)
yelp.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>895</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define a function that accepts text and returns the polarity.
def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity
    #return TextBlob(text).sentiment.polarity
```


```python
# Create a new DataFrame column for sentiment (Warning: SLOW!).
yelp['sentiment'] = yelp.text.apply(detect_sentiment)
```


```python
# Box plot of sentiment grouped by stars
yelp.boxplot(column='sentiment', by='stars')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d7eca58>




![png](output_93_1.png)



```python
# Reviews with most positive sentiment
yelp[yelp.sentiment == 1].text.head()
```




    254    Our server Gary was awesome. Food was amazing....
    347    3 syllables for this place. \r\nA-MAZ-ING!\r\n...
    420                                    LOVE the food!!!!
    459    Love it!!! Wish we still lived in Arizona as C...
    679                                     Excellent burger
    Name: text, dtype: object




```python
# Reviews with most negative sentiment
yelp[yelp.sentiment == -1].text.head()
```




    773     This was absolutely horrible. I got the suprem...
    1517                  Nasty workers and over priced trash
    3266    Absolutely awful... these guys have NO idea wh...
    4766                                       Very bad food!
    5812        I wouldn't send my worst enemy to this place.
    Name: text, dtype: object




```python
# Widen the column display.
pd.set_option('max_colwidth', 500)
```


```python
# Negative sentiment in a 5-star review
yelp[(yelp.stars == 5) & (yelp.sentiment < -0.3)].head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>length</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>390</th>
      <td>106JT5p8e8Chtd0CZpcARw</td>
      <td>2009-08-06</td>
      <td>KowGVoP_gygzdSu6Mt3zKQ</td>
      <td>5</td>
      <td>RIP AZ Coffee Connection.  :(  I stopped by two days ago unaware that they had closed.  I am severely bummed.  This place is irreplaceable!  Damn you, Starbucks and McDonalds!</td>
      <td>review</td>
      <td>jKeaOrPyJ-dI9SNeVqrbww</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>175</td>
      <td>-0.302083</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Positive sentiment in a 1-star review
yelp[(yelp.stars == 1) & (yelp.sentiment > 0.5)].head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>length</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1781</th>
      <td>53YGfwmbW73JhFiemNeyzQ</td>
      <td>2012-06-22</td>
      <td>Gi-4O3EhE175vujbFGDIew</td>
      <td>1</td>
      <td>If you like the stuck up Scottsdale vibe this is a good place for you. The food isn't impressive. Nice outdoor seating.</td>
      <td>review</td>
      <td>Hqgx3IdJAAaoQjvrUnbNvw</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>119</td>
      <td>0.766667</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reset the column display width.
pd.reset_option('max_colwidth')
```

<a id='add_feat'></a>
## Bonus: Adding Features to a Document-Term Matrix

Here, we will add additional features to our `CountVectorizer()`-generated feature set to hopefully improve our model.

To make the best models, you will want to supplement the auto-generated features with new features you think might be important. After all, `CountVectorizer()` typically lowercases text and removes all associations between words. Or, you may have metadata to add in addition to just the text.

> Remember: Although you may have hundreds of thousands of features, each data point is extremely sparse. So, if you add in a new feature, e.g., one that detects if the text is all capital letters, this new feature can still have a huge effect on the model outcome!


```python
# Create a DataFrame that only contains the 5-star and 1-star reviews.
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# define X and y
feature_cols = ['text', 'sentiment', 'cool', 'useful', 'funny']
X = yelp_best_worst[feature_cols]
y = yelp_best_worst.stars

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
# Use CountVectorizer with text column only.
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train.text)
X_test_dtm = vect.transform(X_test.text)
print(X_train_dtm.shape)
print(X_test_dtm.shape)
```

    (3064, 16825)
    (1022, 16825)
    


```python
# Shape of other four feature columns
X_train.drop('text', axis=1).shape
```




    (3064, 4)




```python
# Cast other feature columns to float and convert to a sparse matrix.
extra = sp.sparse.csr_matrix(X_train.drop('text', axis=1).astype(float))
extra.shape
```




    (3064, 4)




```python
# Combine sparse matrices.
X_train_dtm_extra = sp.sparse.hstack((X_train_dtm, extra))
X_train_dtm_extra.shape
```




    (3064, 16829)




```python
# Repeat for testing set.
extra = sp.sparse.csr_matrix(X_test.drop('text', axis=1).astype(float))
X_test_dtm_extra = sp.sparse.hstack((X_test_dtm, extra))
X_test_dtm_extra.shape
```




    (1022, 16829)




```python
# Use logistic regression with text column only.
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))
```

    0.917808219178
    


```python
# Use logistic regression with all features.
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm_extra, y_train)
y_pred_class = logreg.predict(X_test_dtm_extra)
print(metrics.accuracy_score(y_test, y_pred_class))
```

    0.922700587084
    

<a id='more_textblob'></a>
## Bonus: Fun TextBlob Features


```python
# Spelling correction
TextBlob('15 minuets late').correct()
```




    TextBlob("15 minutes late")




```python
# Spellcheck
Word('parot').spellcheck()
```




    [('part', 0.9929478138222849), (u'parrot', 0.007052186177715092)]




```python
# Definitions
Word('bank').define('v')
```




    [u'tip laterally',
     u'enclose with a bank',
     u'do business with a bank or keep an account at a bank',
     u'act as the banker in a game or in gambling',
     u'be in the banking business',
     u'put into a bank account',
     u'cover with ashes so to control the rate of burning',
     u'have confidence or faith in']




```python
# Language identification
TextBlob('Hola amigos').detect_language()
```




    u'es'



<a id="bayes"></a>

## Appendix: Intro to Naive Bayes and Text Classification

Later in the course, we will explore in-depth how to use the Naive Bayes classifier with text. Naive Bayes is a very popular classifier because it has minimal storage requirements, is fast, can be tuned easily with more data, and has found very useful applications in text classificaton. For example, Paul Graham originally proposed using Naive Bayes to detect spam in his [Plan for Spam](http://www.paulgraham.com/spam.html).

Earlier we experimented with text classification using a Naive Bayes model. What exactly are Naive Bayes classifiers? 

**What is Bayes?**  
Bayes, or Bayes' Theorem, is a different way to assess probability. It considers prior information in order to more accurately assess the situation.

**Example:** You are playing roulette.

As you approach the table, you see that the last number the ball landed on was Red-3. With a frequentist mindset, you know that the ball is just as likely to land on Red-3 again given that every slot on the wheel has an equal opportunity of 1 in 37.

However, it is against your intuition to bet on red on the next roll. You think that because it was red this time it is more likely to be black next time. You don't know why, but in the back of your mind you believe that the ball is more likely to land on black given it landed on red previously than it is to land on red twice in a row.

This is what Bayes is all about — adjusting probabilities as more data is gathered!

Below is the equation for Bayes.  

$$P(A \ | \ B) = \frac {P(B \ | \ A) \times P(A)} {P(B)}$$

- **$P(A \ | \ B)$** : Probability of `Event A` occurring given `Event B` has occurred.
- **$P(B \ | \ A)$** : Probability of `Event B` occurring given `Event A` has occurred.
- **$P(A)$** : Probability of `Event A` occurring.
- **$P(B)$** : Probability of `Event B` occurring.



## Applying Naive Bayes Classification to Spam Filtering

Let's pretend we have an email with three words: "Send money now." We'll use Naive Bayes to classify it as **ham or spam.** ("Ham" just means not spam. It can include emails that look like spam but that you opt into!)

$$P(spam \ | \ \text{send money now}) = \frac {P(\text{send money now} \ | \ spam) \times P(spam)} {P(\text{send money now})}$$

By assuming that the features (the words) are conditionally independent, we can simplify the likelihood function:

$$P(spam \ | \ \text{send money now}) \approx \frac {P(\text{send} \ | \ spam) \times P(\text{money} \ | \ spam) \times P(\text{now} \ | \ spam) \times P(spam)} {P(\text{send money now})}$$

Note that each conditional probability in the numerator is easily calculated directly from the training data!

So, we can calculate all of the values in the numerator by examining a corpus of spam email:

$$P(spam \ | \ \text{send money now}) \approx \frac {0.2 \times 0.1 \times 0.1 \times 0.9} {P(\text{send money now})} = \frac {0.0018} {P(\text{send money now})}$$

We would repeat this process with a corpus of ham email:

$$P(ham \ | \ \text{send money now}) \approx \frac {0.05 \times 0.01 \times 0.1 \times 0.1} {P(\text{send money now})} = \frac {0.000005} {P(\text{send money now})}$$

All we care about is whether spam or ham has the higher probability, and so we predict that the email is spam.


### Key Takeaways

- The "naive" assumption of Naive Bayes (that the features are conditionally independent) is critical to making these calculations simple.
- The normalization constant (the denominator) can be ignored since it's the same for all classes.
- The prior probability is much less relevant once you have a lot of features.

### Comparing Naive Bayes With Other Models

Advantages of Naive Bayes:

- Model training and prediction are very fast.
- It's somewhat interpretable.
- No tuning is required.
- Features don't need scaling.
- It's insensitive to irrelevant features (with enough observations).
- It performs better than logistic regression when the training set is very small.

Disadvantages of Naive Bayes:

- If "spam" is dependent on non-independent combinations of individual words, it may not work well.
- Predicted probabilities are not well calibrated.
- Correlated features can be problematic (due to the independence assumption).
- It can't handle negative features (with Multinomial Naive Bayes).
- It has a higher "asymptotic error" than logistic regression.

-----

<a id='conclusion'></a>
## Conclusion

- NLP is a gigantic field.
- Understanding the basics broadens the types of data you can work with.
- Simple techniques go a long way.
- Use scikit-learn for NLP whenever possible.

While we used SKLearn and TextBlob today, another popular python NLP library is [Spacy](https://spacy.io).
