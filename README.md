# Unsupervised Keyword Extraction Using Spatial Analyses

Welcome to the **Unsupervised Keyword Extraction Using Spatial Analyses** repository! This project focuses on extracting keywords from single texts using various unsupervised methods. The repository includes implementations/usages of four methods: Luhn, YAKE, RAKE, and our novel approach. Additionally, we provide a dataset specifically collected for this research.

## Methods Implemented

1. **Luhn**: A classic keyword extraction method based on the frequency and distribution of words. [Read the paper](https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf)
2. **YAKE**: Yet Another Keyword Extractor, which relies on statistical characteristics of text. [Read the paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025519308588?via%3Dihub)
3. **RAKE**: Rapid Automatic Keyword Extraction, a straightforward and efficient algorithm based on word co-occurrence. [Read the paper](https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents)
4. **Our Method**: A new approach leveraging spatial analyses of words within texts to identify key terms. 

## Dataset

We have curated a unique dataset for evaluating the performance of these keyword extraction methods. The dataset can be accessed [here](https://github.com/LidaAleksanyan/keywords_extraction_data/tree/master).

## Installation

To install and run the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/LidaAleksanyan/spatial_keyword_extraction.git
    cd spatial_keyword_extraction
    ```

2. **Install the required Python packages**:
    ```sh
    pip3 install -r requirements.txt
    ```

## Usage

After installation, you can start using the implemented methods to extract keywords from your texts. All method will be executed through the provided scripts, just comment those methods which you don't want to run in main.py file.

Example usage:
```sh
python3 main.py -l english -k 150 -b examples/anna_karenina.txt

