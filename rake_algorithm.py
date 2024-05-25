import RAKE
import operator

def Sort_Tuple(tup):
    tup.sort(key = lambda x: x[1])
    return tup 

def rake(text, top_k=100, stopwords_file_path="stop_words_english.txt"):
    """Usage of keywords extraction algorithm RAKE, codes are taken from here
       https://pypi.org/project/python-rake/

    :param str text: row text, without cleaning it from punctuation, stopwords etc.
    :param int top_k: top_k keywords to print
    :Param str stopwords_file_path: path to stopwords list file
    """
    rake_object = RAKE.Rake(stopwords_file_path)
    keywords = Sort_Tuple(rake_object.run(text))
    print("==============RAKE============")
    print("THE NUMBER OF EXTRACTED KEYWORDS BY RAKE: ", top_k)
    results = []
    for k in keywords[-top_k:]:
        results.append(k[0])
    print(results)
