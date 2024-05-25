import numpy as np
from collections import Counter, OrderedDict

def luhn(text, num_to_take=100):
    """Implementation of Luhn algorithm described here
       https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf

    :param str text: already preprocessed text (without digits, punctuations, stopwords, lemmatized etc.)
    :param int num_to_take: how many extracted words to return
    """
    frequency = OrderedDict({})
    text = np.array(text.split())
    N, i = len(text), 0

    for word in text:
        count = frequency.get(word , 0)
        frequency[word] = count + 1
    sorted_dict = OrderedDict({})
    sorted_keys = sorted(frequency, key=frequency.get, reverse=True)

    for w in sorted_keys:
        sorted_dict[w] = frequency[w]
    res_dict = Counter(sorted_dict.values())
    max_rank = None
    freq = 1
    for k, v in sorted_dict.items():
        max_rank = len(frequency.keys())
        if k not in frequency.keys():
            print("skipping ", k)
            continue
        if res_dict[freq] >= 10:
            for k1, v1 in list(sorted_dict.items()):
                if v1 == freq:
                    frequency.pop(k1, "no such value")
        else:
            break
        freq+=1
    
    luhn_results = sorted(frequency, key=frequency.get, reverse=True)[:num_to_take]
    print("==============LUHN============")
    print("THE NUMBER OF EXTRACTED KEYWORDS BY LUHN: ", len(luhn_results))
    print(luhn_results)
