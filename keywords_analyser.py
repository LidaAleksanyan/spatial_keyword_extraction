import os
import re
import time
import summa 
import gensim
import string
import PyPDF2
import pandas as pd
import numpy as np
from summa import keywords

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from pymystem3 import Mystem
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

class KeywordsAnalyzer():
    def __init__(self, language):
        self.language = language
        self.text = ''
        self.words = []
        self.spatial_freqs = {}
        self.freqs = {}
        self.spatial_moment2 = {}
        self.spatial_moment4 = {}
        self.spatial_moment6 = {}
        self.freqs_list = []
        self.spatial_freq_list = []
        self.spatial_moment2_list = []
        self.spatial_moment4_list = []
        self.spatial_moment6_list = []
    
    def read_book(self, book_path):
        """Reads book which can be in one of PDF or txt formats.

        :param str book_path: full path to book file location
                                    works only for english texts
        """
        print("Reading: ", book_path)
        # PDF files reading
        if book_path[-1] == 'f':
            pdfFileObj = open(book_path, 'rb')
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            text_ = ''
            for n in range(pdfReader.numPages):
                pageObj = pdfReader.getPage(n)
                text_ += pageObj.extractText()
            for n in range(pdfReader.numPages):
                pageObj = pdfReader.getPage(n)
                text_ += pageObj.extractText()
            pdfFileObj.close()
        # Simple txt format files reading
        else:
            with open(book_path, encoding = "utf8", errors = 'ignore') as f:
                text_ = f.read()
        if self.language == "english":
            print("English text")
            self.text = self.preprocess_english_text(text_)
        elif self.language == "russian":
            print("Russian text")
            self.text = self.preprocess_russian_text(text_)
        elif self.language == "french":
            print("French text")
            self.text = self.preprocess_french_text(text_)
    
    def preprocess_english_text(self, text, stopwords_file = "./stopwords/stop_words_english.txt", with_stopwords=False):
        """Basic preprocessing of english text, like stopwords removing, lemmatization, 
           removing punctuations and digits and also 1-2 letter words.

        :param str text: 
        :param str stopwords_file: path to stopwords file
        :param bool with_stopwords: whether to remove stopwords or keep them
        """
        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        
        self.full_text = text
        remove_characters = string.punctuation + "‘" + "’" + '”' + '“' + string.digits
        translation_dict = str.maketrans("", "", remove_characters)
        text = str(text)
        text = text.lower()
        text = text.replace("—", " ")
        text = text.translate(translation_dict)
        text = word_tokenize(text) 
        
        #lemmatizing, then removing stop_words if with_stopwords=False else keep them, can be used for RAKE
        lem = WordNetLemmatizer()
        if with_stopwords:
            text = [lem.lemmatize(word, get_wordnet_pos(word)) for word in text]
        else:
            stopwords = []
            with open(stopwords_file) as f:
                sws = f.readlines()
                for line in sws:
                    stopwords.append(line.split('\n')[0])
            text = [lem.lemmatize(word, get_wordnet_pos(word)) for word in text if word not in stopwords]
        #removing one or two-letter words 
        text = [word for word in text if len(word) > 2]
        text = " ".join(text) 
        
        return text

    def preprocess_russian(self, text):
        """Basic preprocessing of russian text, like stopwords removing, lemmatization, 
           removing punctuations and digits and also 1-2 letter words.

        :param str text: 
        """
        text = str(text) 
        text = text.lower()
        stopwords_russian = stopwords.words('russian')
        additional_stopwords = ["еще", "ещё", "меж", "зато", "пусть", "ага", "этот", "это", "почему",
                        "весь", "ты", "он", "она", "они", "оно", "мы", "вы", "кто", "что",
                        "сам", "сама", "само", "свой", "наш", "ваш", "их", "тот", "та", "те",
                        "то", "раз", "твой", "мой", "кой", "кое", "все", "весь", "всё", "быть", "тот",
                        "таки", "такой", "какой", "каждый", "который", "сказал", "сказала", "своей", 
                        "свое", "очень", "мог", "могла", "могли", "которая", "которые", "которое", 
                        "тебе", "тебя", "могу", "могла", "можешь", "можете", "свое", "своя", "свои", 
                        "эта", "эти", "те", "чтото", "ктото", "этим", "своим", "тех", "всетаки", "кроме", 
                        "собой", "тобой", "своем", "этому", "по-моему", "ну-с", "", "такое", "такая", "такие", 
                        "которых", "очень", "пред", "могу", "и", "а", "в", "б", "д","е", "ж", "з", "к", "л", "м", 
                        "н", "о", "п", "р", "с", "у", "ф", "ч", "ц", "ш", "щ", "ь", "ъ","э", "ю", "я"]
        stopwords = stopwords_russian + additional_stopwords
        mystem = Mystem()
        punctuation = string.punctuation
        punctuation += '–—"¿¡”“„( “, ( (« ) –,'
        tokens = mystem.lemmatize(text)
        tokens = [token for token in tokens if token not in stopwords and token != " " \
                  and token.strip() not in punctuation and token[0] not in string.digits \
                  and len(token) > 2]
        text = " ".join(tokens)
         
        return text
   
    def preprocess_french(self, text, stopwords_file='./stopwords/french_stopwords.txt'):
        """Basic preprocessing of french text, like stopwords removing, lemmatization, 
           removing punctuations and digits and also 1-2 letter words.

        :param str text:
        :param str stopwords_file: path to french stopwords file
        """
        text = str(text) 
        text = text.lower()
        stopwords = []
        with open(stopwords_file) as f:
            sws = f.readlines()
            for line in sws:
                stopwords.append(line.split('\n')[0])
        punctuation = string.punctuation
        punctuation += '––—"¿¡”“„('
        lemmatizer = FrenchLefffLemmatizer()
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords \
                  and token != " " and token.strip() not in punctuation \
                  and token[0] not in string.digits and len(token) > 2]
        text = " ".join(tokens)

        return text
    
    def calc_statistics(self, permute = False):
        #Split text and keep unique words
        text = np.array(self.text.split())
        N, i = len(text), 0
        set_text = set(text)

        if permute:
            text = np.random.RandomState(seed=42).permutation(text)
        
        for w in set_text:
            #find indexes of w occurences in arr
            occurences = np.where(text == w)[0]
            frequency = occurences.size
            if frequency <= 2:
                continue
            sum_second_moment, sum_forth_moment, sum_sixth_moment = 0, 0, 0
            #count of words before first occurence of w and last occurence of w
            st_count, end_count = occurences[0], N - occurences[-1] - 1
            n_words = N - st_count - end_count - frequency
            
            t_w = (n_words + frequency - 1) / (frequency - 1)
            self.spatial_freqs[w] = 1 / t_w # tao(w)
            
            self.freqs[w] = frequency / N
            
            for i in range(1, frequency):
                qsi_i = occurences[i] - occurences[i - 1]
                sum_second_moment += (qsi_i + 1)**2 
                sum_forth_moment += ((qsi_i + 1)**4 / 10000000)
                sum_sixth_moment += ((qsi_i + 1)**6 / 10000000)
            self.spatial_moment2[w] = 1 / (sum_second_moment / (frequency - 1))
            self.spatial_moment4[w] = 1 / (sum_forth_moment / (frequency - 1))
            self.spatial_moment6[w] = 1 / (sum_sixth_moment / (frequency - 1))
        
        self.words = list(self.freqs.keys())
        self.spatial_freq_list = list(self.spatial_freqs.values())
        self.freqs_list = list(self.freqs.values()) 
        self.spatial_moment2_list = list(self.spatial_moment2.values())
        self.spatial_moment4_list = list(self.spatial_moment4.values())
        self.spatial_moment6_list = list(self.spatial_moment6.values())
    
    def run(self, permute = False):
        self.calc_statistics(permute = permute)
        df_space_freq = pd.DataFrame([{
                        "word": w, 
                        "frequency": f,
                        "spatial_freq": sf, 
                        "spatial_moment2": sm2, 
                        "spatial_moment4": sm4, 
                        "spatial_moment6": sm6}
                        for w, f, sf, sm2, sm4, sm6 in zip(self.words,
                                                           self.freqs_list, 
                                                           self.spatial_freq_list, 
                                                           self.spatial_moment2_list,
                                                           self.spatial_moment4_list,
                                                           self.spatial_moment6_list)])
        
        df_space_freq = df_space_freq.sort_values("frequency", ascending = False)
        
        return df_space_freq

def get_text(book_path, language='english'):
    keywords_analyzer_obj = KeywordsAnalyzer(language = language)
    keywords_analyzer_obj.read_book(book_path)
    return keywords_analyzer_obj.full_text, keywords_analyzer_obj.text

def get_results(book_path, language='english', top_k=10, large_text=True):
    keywords_analyzer_non_perm = KeywordsAnalyzer(language = language)
    t = time.time()
    keywords_analyzer_non_perm.read_book(book_path)
    keywords_analyzer_perm = KeywordsAnalyzer(language = language)
    keywords_analyzer_perm.text = keywords_analyzer_non_perm.text
    keywords_analyzer_perm.language = keywords_analyzer_non_perm.language
    print("TIME FOR BOOK READING: ", time.time() - t)
    t = time.time()
    df_space_freq = keywords_analyzer_non_perm.run(permute=False)
    df_space_freq_permuted = keywords_analyzer_perm.run(permute=True)
    print("TIME FOR ALGORITHM: ", time.time() - t)
    rank = np.arange(df_space_freq.shape[0]) + 1

    spatial_moment2_global = list(np.array(keywords_analyzer_perm.spatial_moment2_list) / np.array(keywords_analyzer_non_perm.spatial_moment2_list))
    spatial_moment2_local = list(np.array(keywords_analyzer_non_perm.spatial_moment2_list) / np.array(keywords_analyzer_perm.spatial_moment2_list))
    spatial_moment4_global = list(np.array(keywords_analyzer_perm.spatial_moment4_list) / np.array(keywords_analyzer_non_perm.spatial_moment4_list))
    spatial_moment4_local = list(np.array(keywords_analyzer_non_perm.spatial_moment4_list) / np.array(keywords_analyzer_perm.spatial_moment4_list))
    spatial_moment6_global = list(np.array(keywords_analyzer_perm.spatial_moment6_list) / np.array(keywords_analyzer_non_perm.spatial_moment6_list))
    spatial_moment6_local = list(np.array(keywords_analyzer_non_perm.spatial_moment6_list) / np.array(keywords_analyzer_perm.spatial_moment6_list))

    df_statistics = pd.DataFrame([
                {
                    "word": w, 
                    "frequency": fq, 
                    "spatial_moment2_global": vdl, 
                    "spatial_moment2_local": vdu,
                    "spatial_moment4_global": gvd4l, 
                    "spatial_moment4_local": gvd4u, 
                    "spatial_moment6_global": gvd6l, 
                    "spatial_moment6_local": gvd6u, 
                }
                for w, fq, vdl, vdu, gvd4l, gvd4u, gvd6l, gvd6u in zip(keywords_analyzer_perm.words, 
                                                                       keywords_analyzer_perm.freqs_list, 
                                                                       spatial_moment2_global,
                                                                       spatial_moment2_local,
                                                                       spatial_moment4_global, 
                                                                       spatial_moment4_local, 
                                                                       spatial_moment6_global,
                                                                       spatial_moment6_local)])
    
    df = pd.merge(df_space_freq, df_space_freq_permuted, on="word")
    if not os.path.exists("./output"):
        os.makedirs("./output")
        print("Created output directory for logs.")
    df.to_csv("./output/" + book_path.split('/')[-1].split('.')[0] + '_full.csv') 
    df_statistics = df_statistics.sort_values("frequency", ascending = False)
    df_statistics.to_csv("./output/" + book_path.split('/')[-1].split('.')[0] + '_picks.csv') 
    
    df_statistics3000 = df_statistics[:3000]
    if large_text:
        df_global = df_statistics3000[df_statistics3000.spatial_moment2_global >= 5]
        df_global2 = df_statistics3000[(df_statistics3000.spatial_moment2_global >= 3) & (df_statistics3000.spatial_moment2_global < 5)]
        df_local = df_statistics3000[df_statistics3000.spatial_moment2_local >= 5]
        df_global = df_global.sort_values("frequency", ascending = False)
        df_global2 = df_global2.sort_values("frequency", ascending = False)
        df_local = df_local.sort_values("frequency", ascending = False)
        print("====================================================================")
        print("THE NUMBER OF EXTRACTED KEYWORDS BY OUR METHOD: ", len(list(df_global.word)) + len(list(df_global2.word)) + len(list(df_local.word)))
        print("GLOBAL KEYWORDS===================================================")
        print("***STRONG CASES***")
        print(list(df_global.word)[:top_k])
        print("***WEAKER CASES***")
        print(list(df_global2.word)[:top_k])
        print()
        print("LOCAL KEYWORDS====================================================")
        print(list(df_local.word)[:top_k])
    else:
        df_global_moment6 = df_statistics3000[df_statistics3000.spatial_moment6_global >= 3]
        df_local_moment6 = df_statistics3000[df_statistics3000.spatial_moment6_local >= 3]
        print("GLOBAL KEYWORDS===================================================")
        print(list(df_global_moment6.word)[:top_k])
        print("LOCAL KEYWORDS====================================================")
        print(list(df_local_moment6.word)[:top_k])
