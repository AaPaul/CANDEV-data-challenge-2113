from rake_nltk import Rake

r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

policy = "Public servants must understand what an access to information request is in order to be able to process it properly while protecting personal information. This course describes the steps to follow when a request for access to information is received and explains how to process and protect personal information by exploring various scenarios. "

r.extract_keywords_from_text(policy)

print(r.get_ranked_phrases()) # To get keyword phrases ranked highest to lowest.