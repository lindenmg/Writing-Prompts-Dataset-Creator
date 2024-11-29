Writing Prompts Short Stories Dataset Creator
=============================================

This Project from 2019 supplies preprocessed short stories for further training or processing with language models.

The Writing Prompts subreddit provides for users the opportunity to write short stories - mostly between 100 and 1000 words.
The stories are based on a short prompt given by another user, whereby the story authors should freely follow the prompt.
The subreddit contains a diverse and very numerous collection of short stories, making a good data source for Deep Neural Network language models.

Note, that this dataset preprocessing has been created before the advent of modern large language models (LLMs).
These LLMs should be capable of writing stories at least as good as found in the Writing Prompts subreddit.

The code provides functionality for the following:
1. It takes in raw data from the Writing Prompts Subreddit (originally) downloaded via Reddit API
2. Cleans the text by standardizing whitespaces, special characters, Unicode and some words
3. Prunes the data by removing most unwanted parts from the short stories and prompts and 
   nonsense text, stories with strong profanity, non-English text etc.
4. Filters the data by removing outliers based on simple statistics
5. Tokenizes the stories and prompts with a pretrained tokenizer of a BERT family model (e.g. RoBERTa)
6. Encodes the sentences and prompts with a pretrained BERT family model


## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Data Preparation](#Writing-Prompts-data-preparation)
3. [Execution of Preprocessing](#execution-of-preprocessing)
4. [Project Organization](#project-organization)


## Setup Instructions

### Python Environment
- Python 3.6 is recommended
- Using a virtual environment is recommended.
- Install PyTorch manually to choose the correct version for your system. 
  Look into requirements.txt for further instructions
- Install the Python requirements with `pip install -r requirements.txt`.
- If issues arise with pyfasttext - "module Cython not found" -, install Cython manually beforehand
- PyTest is used for unit tests.

### Config File
1. Copy `config.json_template` and rename the copy to `config.json`. 
2. Replace the placeholder `<project_root_path>` with the absolute path to the projects root directory
3. Add the corresponding values to each key in the file that fits your desired configuration: 
   - In the section *hardware* set the appropriate values for CPU cores and turn CUDA GPU usage on or off
   - *path.pretrained_model* determines which BERT or GPT family model from Huggingface. [Reference](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodel)

### Third Party Files
- Download [lid.176.bin](https://fasttext.cc/docs/en/language-identification.html) (126MB) and save it in data/external.
  It is a simple (and error-prone) model for detecting the given language (e.g. English) of a text.
  It falls under the [Creative Commons Attribution-Share-Alike License 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- Download the [glove.840B.300d.zip GloVe word vector file](https://nlp.stanford.edu/projects/glove/) (2GB zipped) 
  and save it also in **unzipped (unpacked) form** in data/external
- Open a Python command line and execute
  ```
  import nltk
  nltk.download('punkt')
  ```
  This downloads the NLTK English sentence tokenizer (sentence boundary detector)
- For spaCy, after installation, activate your virtual environment and run:
  ```
  python -m spacy download en
  ```


## Writing Prompts Data Preparation
Just **use the old** Writing Prompts **dataset** which is stored in `data/raw` via [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) as `Writing Prompts.db`. Decompress/Extract it before usage!

### Download Subreddit Data
- **These instructions are deprecated!** 
  Reddit has changed its API usage conditions. You can now download data only very slowly unless you want to pay.
- Subreddit data download:
    - `git clone https://github.com/voussoir/timesearch` into *src/*. Follow instructions of its Readme
    - The timesearch subreddit "crawl" might take quite some time until it starts
    - python3 timesearch.py timesearch -r Writing Prompts -l update
    - python3 timesearch.py commentaugment -r Writing Prompts -l update
    - Latest time for thread submissions: Jul 12 2019 10:54:56 +76
- Move sqlite database from timesearch subreddit crawl to <path_to_project_root>/data/raw/Writing Prompts.db
- Create symbolic link so data is in data/raw but can also be updated by timesearch:
    - ln -sn <path_to_project_root>/data/raw/Writing Prompts.db src/timesearch/subreddits/Writing Prompts/Writing Prompts.db

## Execution of Preprocessing
After following the sections *Setup Instructions* and *Data Preparation* execute the Jupyter Notebooks in notebooks/.
Execute the notebooks in order from 01 to 05. Read the comments in 04 & 05 for the correct encoding settings.
If you want to use the data, follow the code from/behind notebooks 04 & 05 to learn how to load the preprocessed stories from the SQLite database 
or the encoded ones from HDFS.


## Project Organization

**File Overview:**
```
├── config            <== Configuration template and file (after setup)
├── data              <== Data in different preprocessing stages & third party data
├── LICENSE          
├── models            <== Where the pretrained model from Huggingface is stored
├── notebooks         <== Jupyter Python Notebooks. Execution of preprocessing, encoding & exploration.
├── README.md        
├── requirements.txt  <== Python Pip packages requirements. See chapter 'Setup'
├── src               <== The source code (except Jupyter Notebooks).
└──  test             <== Unit tests
```

**src/ Code Base:**
```
├── data
│   ├── data_cleaning.py     <== Normalizing text, words and characters
│   ├── data_exploration.py  <== Manual exploration. For finding story/prompt preprocessing targets
│   ├── data_filtering.py    <== Filters out unwanted text
│   ├── data_pruning.py      <== Removes unwanted parts in text
│   ├── spacyoverlay.py      <== For easier usage of Spacy package
│   └── vocab_coverage.py    <== To determine how many words of the word vector vocabulary are covered
├── preprocessing
│   ├── dataset_creation.py    <== Create tokenized dataset, split text also into its sentences
│   ├── text_encoding.py       <== Encode text with BERT-style model
│   └── token_encoding.py      <== Encoding of text with trained tokenizer
└── utils
    ├── data_processing.py  <== Simple data structure transform functions
    ├── hdfs_caching.py     <== Save and load data from/to HDFS (Hadoop distributed file system)
    ├── helpers.py          <== Miscallenous helper functions
    ├── settings.py         <== Accessing config.json file
    └── sqlite.py           <== Sqlite database key-value store functionality
```

--------

<p><small>Project folder structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
