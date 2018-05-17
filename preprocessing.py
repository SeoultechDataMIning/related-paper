import nltk
''' remove special symbols & to lower case'''
import re
def remove_special_symbol(string):
    letters = re.sub('[^a-zA-Z]', ' ', string)
    return letters
def to_lower_case(string):
    lower_string = string.lower()
    return lower_string
def remove_spesym_tolower(string):
    letters = remove_special_symbol(string)
    lower_string = letters.lower()
    return lower_string

'''tokenize string: 토큰화'''
from nltk.tokenize import TweetTokenizer
def tokenize_corpus(string):
    tknizer = TweetTokenizer()
    tokens = tknizer.tokenize(string)
    return tokens

'''remove stopwords: 불용어'''
from nltk.corpus import stopwords
def remove_stopwords(tokens):
    words = [w for w in tokens if not w in stopwords.words('english')]
    return words

'''stemming tokens: 어간추출'''
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
def porter_stemmer(tokens):
    porter = nltk.PorterStemmer()
    words = [porter.stem(t) for t in tokens]
    return words
def lancaster_stemmer(tokens):
    lancaster = nltk.LancasterStemmer()
    words = [lancaster.stem(t) for t in tokens]
    return words
def snowball_stemmer(tokens):
    snowball = nltk.SnowballStemmer()
    words = [snowball.stem(t) for t in tokens]
    return words
'''lemmatization: 음소표기법'''
from nltk.stem import WordNetLemmatizer
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in tokens]
    return words


def normalizer(corpus):
    # 1. remove special symbols and convert to lowercase
    string = remove_spesym_tolower(corpus)

    # 2. tokenize
    tokens = tokenize_corpus(string)
    
    # 3. remove stopwords
    tokens = remove_stopwords(tokens)

    #4. stemming..  choose stemmer among Porter, lancaster, snowball and lemmatizer
    tokens = porter_stemmer(tokens)

    return ( ' '.join(tokens))

