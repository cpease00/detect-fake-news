# Fake News Detection Classification Algorithm
Check out this [Medium post](https://towardsdatascience.com/machine-learning-tackles-the-fake-news-problem-c3fa75549e52) for more details

 ## The Solution to a Problem
 There is a vast quantity of misleading and false news articles spread throughout social media. This leads to a lot of misplaced fear, animosity, and recently attempts to sway political elections. It is very hard to detect 'fake news' when you see it, and there are too many articles to effectively combat their spread by eye. Machine learning techniques offer a clear, scalable solution for identification of malicious news articles. In this project the problem is tackled as a binary classification, using natural language processing (NLP) and supervised learning methods. Using a [Kaggle dataset](https://www.kaggle.com/c/fake-news) of articles from wide-ranging sources, a Naïve Bayes Classifier and Logistic Regression achieved a testing accuracy of 94%.
 
## Methodology
The task at hand is of Natural Language Classification, which approached as a supervised machine learning problem. The first step was to clean and vectorize the data to enable training of the algorithm. Scikit-learn makes it very easy to vectorize each article. The process is to create a bag of words, or corpus combining all the words that appear across all articles. Then those words which appear most commonly across the corpus can be used to assign vectors representing to each article (either by count, frequency, or other methods).

Term frequency/inverse document frequency (TF-IDF) was chosen for the vectorization, as it lends itself to the problem by identifying text patterns that are infrequent in the corpus as a whole. The vectorizer also allows for custom preprocessing and tokenizing functions, in which we utilized regular expressions to parse out undesirable characters and stop words. The tokenizer allowed for uni- as well as bi-grams in the vectorization process.

![ArticleVector](https://github.com/cpease00/detect-fake-news/blob/master/vectorized_article.jpg)

This is an example of the vector associated with one of the articles. Most vectors are more sparse than this example, as shorter articles will not contain many of the feature words.

## Feature Engineering
Once the articles were vectorized, a number of meta-features were created in order to improve the discriminative power of the model. The number of numerals and average word length were chosen to gain sensitivity to different aspects of the article's style with simple calculations. In addition, simple sentiment metrics were calculated using Vader and Textblob to gauge polarity, subjectivity, and neutrality. Althogh these metrics had little difference in their distibution for the two classes, they nonetheless provided improved classification accuracy.

Below is the distribution of sentiment scores assigned to each text, produced by TextBlob:

![Polarity](https://github.com/cpease00/detect-fake-news/blob/master/Sentiment_Polarity_Distribution.jpg)
The general overlap indicates that this variable alone would not suffice to separate the two populations by traditional statistical methods. Below you can see the distribution of subjectivity scores:

![Subjectivity](https://github.com/cpease00/detect-fake-news/blob/master/Subjectivity_Distribution.jpg)

The peak at 0 subjectivity (meaning very objective) for the fake articles may seem counterintuitive. However this makes sense in that misleading articles often represent subjective statements of opinion as objective facts.

## Model Selection
The difference between the two classes is stark at the extreme ends of the spectrum, but there is a large grey area in the middle. To model the probability of an article being reliable, a Multinomial Logistic Regression and Naïve Bayes Classifier were employed, using sklearn. The two represent discriminative and generative approaches respectively, and approached the same results. These two models were selected for their interpretability as well as ease of implementation. 

## Results
In addition to the testing accuracy of 94%, the following ROC curves highlight the sensitivity of the chosen models.
![LogisticRegression](https://github.com/cpease00/detect-fake-news/blob/master/roc_lr.jpg "LogReg ROC and Feature Importances")

It is interesting to observe the differences in important features for each model, as it offers some insight as to how the decisions are being made. For example, 'Hillary', 'Trump' and 'Obama' are indicators of a less trustworthy article, while 'Mr Obama', 'President Trump' and 'Mrs. Clinton' are more likely incuded in reliable sources. This makes sense as a certain level of respect is more commonly assigned to these people's names in traditional news sources.

![NaiveBayes](https://github.com/cpease00/detect-fake-news/blob/master/roc_nb.jpg "NBayes ROC and Feature Importances")

It is clear that this is an effective method for quickly flagging articles that are of dubious reliability. However, the supervised nature of the approach is not representative of the real problem. While this model is very effective at generalizing on outside news data at the moment, over time it will become less powerful unless it is retrained on new labeled articles. The next steps will certainly be to pursue an unsupervised approach, and perhaps implement the model as a browser extension for easy use for individuals. Additional interpretability could be achieved with more interactivity, for example a dashboard with user-input would be great to show what types of writing style and language are more likely to classified as fake. 
