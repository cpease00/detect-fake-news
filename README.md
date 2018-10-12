# Fake News Detection Classification Algorithm
[Medium post](https://towardsdatascience.com/machine-learning-tackles-the-fake-news-problem-c3fa75549e52)
 ## The Solution to a Problem
 There is a vast quantity of misleading and false news articles spread throughout social media. This leads to a lot of misplaced fear, animosity, and recently attempts to sway political elections. It is very hard to detect 'fake news' when you see it, and there are too many articles to effectively combat their spread by eye. Machine learning techniques offer a clear, scalable solution for identification of malicious news articles. In this project the problem is tackled as a binary classification, using natural language processing (NLP) and supervised learning methods. Using a [Kaggle dataset](https://www.kaggle.com/c/fake-news) of articles from wide-ranging sources, a Naïve Bayes Classifier and Logistic Regression achieved a testing accuracy of 94%.
 
## Methodology
The task at hand is of Natural Language Classification, which approached as a supervised machine learning problem. The first step was to clean and vectorize the data to enable training of the algorithm. Scikit-learn makes it very easy to vectorize each article. We decided that term frequency/inverse document frequency (TF-IDF) would lend itself to the problem by identifying text patterns that would infrequent in the corpus as a whole. The vectorizer also allows for custom preprocessing and tokenizing functions, which utilized regular expressions to parse out undesirable characters and stop words. The tokenizer allowed for uni- as well as bi-grams in the vectorization process.

## Feature Engineering
Once the articles were vectorized, a number of meta-features were created in order to improve the discriminative power of the model. The number of numerals and average word length were chosen to gain sensitivity to different aspects of the article's style with simple calculations. In addition, simple sentiment metrics were calculated using Vader and Textblob to gauge polarity, subjectivity, and neutrality. Althogh these metrics had little difference in their distibution for the two classes, they nonetheless provided improved classification accuracy.

## Model Selection
The difference between the two classes is stark at the extreme ends of the spectrum, but there is a large grey area in the middle. To model the probability of an article being reliable, a Multinomial Logistic Regression and Naïve Bayes Classifier were employed, using sklearn. The two represent discriminative and generative approaches respectively, and approached the same results. These two models were selected for their interpretability as well as ease of implementation. 

## Results
