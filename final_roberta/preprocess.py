import json
import nltk
import spacy
import numpy as np
import tqdm
import unidecode
from bs4 import BeautifulSoup
import re
import unicodedata
from itertools import groupby


class SentenceEmbed:
  def __init__(self):
    self.nlp = spacy.load('en_core_web_lg',
                              disable=['parser', 'tagger'])
    
  def embed(self, context):
    vs = []
    v_lens = []
    context = [c.lower() for c in context]
    context_docs = self.nlp.pipe(context, n_process=4)
    for doc in context_docs:
        v = [token.vector for token in doc if token.has_vector]
        v_lens.append(len(v))
        vs.append(np.array(v))
    return vs,v_lens
 
sen_embed = SentenceEmbed()


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

# def stem_words(words):
#     """Stem words in list of tokenized words"""
#     stemmer = LancasterStemmer()
#     stems = []
#     for word in words:
#         stem = stemmer.stem(word)
#         stems.append(stem)
#     return stems
#
# def lemmatize_verbs(words):
#     """Lemmatize verbs in list of tokenized words"""
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas

def normalize(words):
    words = remove_non_ascii(words)
#     words = to_lowercase(words)
    words = remove_punctuation(words)
#     words = replace_numbers(words)
#     words = remove_stopwords(words)
    return words

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def preprocess_sentence(sen):
    sen = sen.strip()
#     sen = re.sub(uri_re, "", sen)
    sen = sen.encode('ascii',errors='ignore')
    #sen = unidecode.unidecode(sen)
    sen = denoise_text(sen)
    # sen = replace_contractions(sen)
    sen = nltk.tokenize.word_tokenize(sen)
    sen = normalize([s for s in sen])
    #print(sen)
#     sen = normalize(unicode(sen))
#     return sen
    return sen
#     tokenized = nltk.tokenize.word_tokenize(temp)
#     final = normalize(unicode(tokenized))
#     return ''.join(final)

# def NER(sen):
#     doc = nlp(unicode(sen))
#     return [d.ent_iob_+'-'+d.ent_type_ if d.ent_iob_ != 'O' else d.text for d in doc ], [d.text for d in doc]
def NER(sen):
    doc = sen_embed.nlp(sen)
#     text = doc.text
#     for ent in doc.ents:
#         text = text.replace(ent.text, ent.label_+'_')
    tokens = [d.text for d in doc]
#     [ent.merge(ent.root.tag_, ent.text, ent.label_) for ent in doc.ents]
#     return compact([d.ent_iob_+'-'+d.ent_type_ if d.ent_iob_ != 'O' else d.text for d in doc ]), tokens
#     return text, tokens
    temp = [d.ent_type_+'_' if d.ent_iob_ != 'O' else d.text for d in doc]
    return [x[0] for x in groupby(temp)], tokens
