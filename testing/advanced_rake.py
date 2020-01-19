from rake_nltk import Metric, Rake

# To use it with a specific language supported by nltk.
r = Rake(language="english")

# If you want to provide your own set of stop words and punctuations to
r = Rake(
    stopwords=["the", "and", "of", "a", "an", "to", "is",
                 "are", "was", "were", "this", "that",
                 "be", "s", "for", "with", "it", "say", 
                 "i", "must", "some"],
    punctuations="!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
)

# If you want to control the metric for ranking. Paper uses d(w)/f(w) as the
# metric. You can use this API with the following metrics:
# 1. d(w)/f(w) (Default metric) Ratio of degree of word to its frequency.
# 2. d(w) Degree of word only.
# 3. f(w) Frequency of word only.

r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
r = Rake(ranking_metric=Metric.WORD_DEGREE)
r = Rake(ranking_metric=Metric.WORD_FREQUENCY)

# If you want to control the max or min words in a phrase, for it to be
# considered for ranking you can initialize a Rake instance as below:

r = Rake(min_length=1, max_length=2)

policy = "Public servants must understand what an access to information request is in order to be able to process it properly while protecting personal information. This course describes the steps to follow when a request for access to information is received and explains how to process and protect personal information by exploring various scenarios. "

r.extract_keywords_from_text(policy)

print(r.get_ranked_phrases()) # To get keyword phrases ranked highest to lowest.