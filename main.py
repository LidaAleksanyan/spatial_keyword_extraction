import os
from tqdm import tqdm
import argparse

import keywords_analyser
from luhn_algorithm import luhn
from yake_algorithm import YAKE
from rake_algorithm import rake

def parse_args():
    parser = argparse.ArgumentParser(description = "Calculate space frequency")
    parser.add_argument('-l', '--language', type=str, required=True,
                     help='text translation language')
    parser.add_argument('-b', '--book', type=str, required=False,
                     help='single text path')
    parser.add_argument('-f', '--txts_folder', type=str, required=False,
                     help='multiple small texta path')
    parser.add_argument('-s', '--stopwords_file', type=str, required=False, default='./stopwords/stop_words_english.txt',
                     help='single text path')
    parser.add_argument('-k', '--top_k', type=int, required=False, default=10,
                     help='top k words to take from each algorithm')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    book_list = os.listdir(args.txts_folder) 
    # This part is for large texts, or for using second degree variance
    # If you want to use single text with six degree variance 
    # turn off large_text parameter in keyword_analyser.get_results function
    if args.book:
        row_text, processed_text = keywords_analyser.get_text(args.book, language=args.language)
        keywords_analyser.get_results(args.book, language=args.language, top_k=args.top_k, large_text=True)
        luhn(processed_text, num_to_take=args.top_k)
        # For YAKE you can give either row_text as it is done in original paper, 
        # or processed text like we do in our research, results are comparable,
        # none of the approaches can be called better than another
        YAKE(processed_text, top_k=args.top_k)
        rake(row_text, top_k=args.top_k, stopwords_file_path=args.stopwords_file)
    # This part is primarily for many small texts in a given folder with six degree variance
    else:
        results = {}
        for book in tqdm(book_list):
            row_text, processed_text = keywords_analyser.get_text(os.path.join(args.txts_folder, book), language=args.language)
            keywords_analyser.get_results(os.path.join(args.txts_folder, book), language=args.language, top_k=args.top_k, large_text=False)
            luhn(processed_text, num_to_take=args.top_k)
            YAKE(processed_text, top_k=args.top_k)
            rake(row_text, top_k=args.top_k, stopwords_file_path=args.stopwords_file)
