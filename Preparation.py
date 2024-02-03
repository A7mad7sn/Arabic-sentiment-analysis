from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import qalsadi.lemmatizer
import string
import re


class Preparation:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def preprocess(self, text):
        punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation

        # Arabic stop words with nltk
        stop_words = stopwords.words("arabic")

        arabic_diacritics = re.compile("""
                                         ّ    | # Shadda
                                         َ    | # Fatha
                                         ً    | # Tanwin Fath
                                         ُ    | # Damma
                                         ٌ    | # Tanwin Damm
                                         ِ    | # Kasra
                                         ٍ    | # Tanwin Kasr
                                         ْ    | # Sukun
                                         ـ     # Tatwil/Kashida
                                     """, re.VERBOSE)
        # remove punctuations
        translator = str.maketrans('', '', punctuations)
        text = text.translate(translator)

        # remove Tashkeel
        text = re.sub(arabic_diacritics, '', text)

        text = ' '.join(word for word in text.split() if word.lower() not in stop_words)

        # Lemmatizing
        lemmer = qalsadi.lemmatizer.Lemmatizer()
        lemmas = lemmer.lemmatize_text(text)
        text = " ".join(lemmas)

        return text

    def feature_extraction_train(self, X):
        print('feature extraction')
        self.tokenizer.fit_on_texts(X)

        sequences = self.tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=4800)
        return padded_sequences

    def feature_extraction_test(self, X):
        print('feature extraction')
        sequences = self.tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=1000)
        return padded_sequences, self.tokenizer

