
#  Copyright Mitra Mirshafiee, 2022 Licensed under MIT License.
#  See the LICENSE.txt for more information.

"""
In this file, you'll find some functions that can 
help you clean your text data.
"""
import regex as re
import string

# punctuation
punc = string.punctuation
t = t.translate(str.maketrans('', '', punc))


# list of some patterns that can be found and substituted in text
rep_character_entity_refrences = {"&gt;": ">", "&lt;":"<",
                                      "&amp;": "&"}

# Go on until you hit a space
t = re.sub(r"\S*https?:\S*", "", t)

# not one of the following punctuations: .-(),
t = re.sub(r'[^ \w\.\-\(\)\,]', ' ', t)

# removes all single letters surrounded with space except letters I and a
t = re.sub(r' +(?![ia])[a-z] +', ' ', t)

# removes all hashtags and the t right after them
t = re.sub(r'[@#]\w*\_*' , '', t)


# substitute extra space with only one space
t = re.sub(r' \s+', ' ', t)

# remove duplicate
t = re.sub(r' \. \.', '\.', t)

# dot at the end and beginning
t = re.sub(' \.$', '\.', t)
t = re.sub('^ *\. *', '', t)
# persian
t = re.sub('[۱۲۳۴۵۶۷۸۹۰]', '', t)

# remove emojis
emojis = re.compile(pattern = "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       "]+", flags = re.UNICODE)

t = emojis.sub(r'',t)


# substitue new lines or tabs with only one space
t = re.sub("\n|\t", ' ', t)



# nltk stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk_stopwords = stopwords.words('english')

# spacy stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = list(nlp.Defaults.stop_words.copy())


words_to_remove =  ['absolutely', "actually"]

total_stopwords = set(
    nltk_stopwords + spacy_stopwords + words_to_remove
    )

# function for removing these words
def remove_st(t):
    return " ".join([i for i in t.split() if i not in total_stopwords])