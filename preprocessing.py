# coding: utf-8

"""Generic text preprocessing."""

import re
import treetaggerwrapper as ttw

from configuration_manager import ConfigurationManager
import lemmatization
import splitting
import my_stopwords

cmgr = ConfigurationManager("config.json")
cmgr.load_config()

language = cmgr.get_default_language()
emotions_array_length = cmgr.get_emotion_array_length()
emotion_names = cmgr.get_emotion_names()

data_dir = cmgr.get_data_dir()

splitter = splitting.TextSplitter(language)
tagger = ttw.TreeTagger(TAGLANG=language.lower()[0:2],
                        TAGDIR=cmgr.get_treetagger_path())
lemmatizer = lemmatization.Lemmatizer(tagger)

all_stopwords = my_stopwords.my_stopwords


def prepare_for_w2v(text, lemmatize=True, keep_stopwords=False,
                    debug=True):
    """Prepare a text for word2vec.

    Parameters
    ----------
    text: str
        the whole text as a single string

    lemmatize: bool
        if True, lemmatize text, otherwise leave it as it is

    keep_stopwords: Bool
        keep or remove stopwords

    Returns
    -------
    cleaned_text: str
        the original text without quotes, multiple spaces, guillemets,
        dashes, apostrophes."""
    # 1. preprocess removing punctuation and other unnecessary glyphs
    preprocessed_text = preprocess(text)

    # 2. split the sentences -> gen(List[List[str]])
    if debug:
        print("Splitting sentences...")
    sentences = (splitter.sentence_to_words(s)
                 for s in splitter.text_to_sentences(preprocessed_text))

    # 3. lemmatize sentences -> gen(List[List[str]])
    if lemmatize:
        if debug:
            print("Start lemmatization...")
        lemmatized = (lemmatizer.lemmatize(s) for s in sentences)
        to_filter_punct = lemmatized
    else:
        to_filter_punct = sentences

    # 4. filter out single punctuation characters from lemmatized
    # -> gen(gen(List[str]))
    punct = {',', '.', ';', ':', '?', '!', '|', '-', '--'}
    if keep_stopwords:
        def fun(w):
            return w not in punct
    else:
        def fun(w):
            return w not in punct and len(w) > 1

    if debug:
        print("Filtering not words...")
    only_words = (filter(fun, s) for s in to_filter_punct)

    # 5. remove apostrophes and dots from remaining words
    # gen(gen(List[str])) -> gen(gen(List[str]))

    def strip_dot_apostrophe(word):
        """Delete dots and apostrophes from a str."""
        return punctuation_to_space(apostrophe_no_space(word))

    if debug:
        print("Stripping apostrophes and dots...")
    cleaned_words = ((strip_dot_apostrophe(w) for w in s)
                     for s in only_words)

    # 6. convert to list
    # gen(gen(List[str])) -> List[List[str]]
    if debug:
        print("Cleaning text...")
    if keep_stopwords:
        cleaned_text = list(map(list, cleaned_words))
    else:
        stopwords = get_stopwords()
        gen_no_stopwords = (filter((lambda w: w not in stopwords), s)
                            for s in cleaned_words)
        if debug:
            print("Deleting stopwords...")
        cleaned_text = list(map(list, gen_no_stopwords))

    # 7. return the List[List[str]] of cleaned_text
    if debug:
        print("Finished!")
    number_of_sentences = len(cleaned_text)
    return cleaned_text, number_of_sentences


def get_stopwords():
    """Get the set of stopwords."""
    return all_stopwords


def preprocess(text):
    """Preprocess a text for tokenization.

    This will substitute all dialogues with fullstop-delimited sentences,
    all \"weak\" punctuation with a fullstop.

    Parameters
    ----------
    text : string
        the text to preprocess

    Returns
    -------
    processed_text : string
        the preprocessed text

    Examples
    --------
    >>> text = "Hello, I am Pietro. The cat is on the table: I don't like it."
    >>> preprocess(text)
    "Hello, I am Pietro. The cat is on the table. I don't like it."
    """
    return rebalance_full_stops(
        ellipsis(
            punctuation(
                multiple_spaces(
                    quotations_in_sentence(
                        guillements_in_sentence(
                            whitespace(
                                end_of_sentence(
                                    apostrophe(
                                        square_brackets(
                                            delete_numbers(text)))))))))))


def delete_numbers(text):
    """Delete all numbers from a text."""
    return re.sub(r'\d', '', text)


def punctuation(text):
    """Substitute ?,!,;,:...- with ."""
    return re.sub(r'[?!;:\|\u2026\u2212\u002d\ufe63\uff0d\u2014\u2013\(\)]+',
                  r'.',
                  text)


def punctuation_to_space(text):
    """Substitute ?,!,;,:...- with a single space."""
    return re.sub(r'[?!;:\|\u2026\u2212\u002d\ufe63\uff0d\u2014\(\)]+',
                  r' ',
                  text)


def ellipsis(text):
    """Substitute ..+ with ."""
    return re.sub(r'[.]{2,}', r'.', text)


def rebalance_full_stops(text):
    """Substitute non consecutive full stops with one only."""
    return re.sub(r'\.(\s*\.*)*', r'. ', text)


def square_brackets(text):
    """Substitute [something] with \"\"."""
    return re.sub(r'\[[^\]]*\]', r'', text)


def multiple_spaces(text):
    """Substitute multiple spaces with a single one, don't touch specials."""
    return re.sub(r'\s{2,}', r' ', text)


def whitespace(text):
    """Substitute all newlines and tabs with a single fullstop."""
    return re.sub(r'[\t\n\r\v\f]+', r'. ', text)


def apostrophe(text):
    """Substitute apostrophe with space."""
    return re.sub(r'[\u0027\u2019\u02bc]+', r' ', text)


def apostrophe_no_space(text):
    """Substitute apostrophe with space."""
    return re.sub(r'[\u0027\u2019\u02bc]+', r'', text)


def guillements_in_sentence(text):
    """Substitute all quotations in a sentence with a space."""
    return re.sub(r'[\u00ab]([^\u00bb]+)[\u00bb]', r' \1 ', text)


def quotations_in_sentence(text):
    """Substitute all opening-closing quotation marks with a space."""
    return re.sub(r'[\u0022]([^\u0022]+)[\u0022]', r' \1 ', text)


def end_of_sentence(text):
    """Put a space after each fullstop."""
    return re.sub(r'\.', r'. ', text)
