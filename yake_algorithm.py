import yake

def YAKE(text, top_k=100):
    """YAKE keywords extracton algorithm taken from here
       https://pypi.org/project/yake/

    :param str text: already preprocessed text (without digits, punctuation, stopwords, 
                                                1-2 letter words, also lemmatized, etc)
    :param int top_k: how many keywords to extract
    """
    kw_extractor = yake.KeywordExtractor(n=1, top=top_k)
    keywords = kw_extractor.extract_keywords(text)
    yake_results = []
    for kw in keywords:
        yake_results.append(kw[0])

    print("==============YAKE============")
    print("THE NUMBER OF EXTRACTED KEYWORDS BY YAKE: ", len(yake_results))
    print(yake_results)
