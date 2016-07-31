import re

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel

from nltk import word_tokenize, pos_tag
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import treebank
from nltk.corpus import wordnet as wn
from nltk.tag import DefaultTagger, UnigramTagger

from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize



def tokenize(text):
    """
    Tokenizes the text by words.

    Parameters
    ----------
    text: A string.

    Returns
    -------
    A list of strings.
    """

    tokens = word_tokenize(text)

    return tokens


def find_best_bigrams(tokens):
    """
    Builds collocations by using the pointwise mutual information (PMI).

    Parameters
    ----------
    tokens: A list of strings.

    Returns
    -------
    A list of tuples of (str, str).
    """

    top_bgs = 10

    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    bigrams = finder.nbest(bigram_measures.pmi, top_bgs)

    return bigrams


def tag_words(words, tag):
    """
    Associates a tag with words.

    Parameters
    ----------
    words: A list of strings.
    tag: A str.

    Returns
    -------
    A list of tuples of (str, str)
    """

    default_tagger = DefaultTagger(tag)
    tags = default_tagger.tag(words)

    return tags


def tag_pos(words):
    """
    Creates Part of Speech tags.

    Parameters
    ----------
    words: A list of strings.

    Returns
    -------
    A list of tuples of (str, str)
    """

    pos_tags = pos_tag(words)

    return pos_tags


def tag_penn(words):
    """
    Tokenizes text by using a Penn Treebank tagged sentence and word tokenizer.

    Parameters
    ----------
    words: A list of strings.

    Returns
    -------
    A list of tuples of (str, str)
    """

    pt_tagger = UnigramTagger(treebank.tagged_sents())
    tags = pt_tagger.tag(words)

    return tags


def tag_linked(words, default_tag='INFO'):
    """
    Tokenizes text by using a Penn Treebank tagged sentence and word tokenizers.
    Uses DefaultTagger to assign "default_tag" to any element missed by Penn Treebank tagger.

    Parameters
    ----------
    words: A list of strings.

    Returns
    -------
    A list of tuples of (str, str)
    :param default_tag:
    """

    default_tagger = DefaultTagger(default_tag)
    pt_tagger = UnigramTagger(treebank.tagged_sents())

    pt_tagger._taggers = [pt_tagger, default_tagger]

    tags = pt_tagger.tag(words)

    return tags


def extract_tags(words):
    """
    Restricts tokens in the text to Nouns, Verbs, Adjectives, and Adverbs.

    Parameters
    ----------
    words: A list of strings.

    Returns
    -------
    A tuple of (pos_tags, terms)
    pos_tags: A list of tuples of (str, str).
    terms: A list of strings.
           Terms extracted with regex.
           Nouns, verbs, adjectives, or adverbs.
    """

    rgxs = re.compile(r"(JJ|NN|VB|RB)")

    pos_tags = pos_tag(words)
    terms = [tkn[0] for tkn in pos_tags if re.match(rgxs, tkn[1])]

    return pos_tags, terms


def get_document_term_matrix(train_data, test_data):
    """
    Uses TfidfVectorizer to create a document term matrix for "train_data" and "test_data".

    Paramters
    ---------
    train_data: A list of strings
    test_data:A list of strings

    Returns
    -------
    A 3-tuple of (model, train_matrix, test_matrix).
    model: A TfidfVectorizer instance
    train_matrix: A scipy.csr_matrix
    test_matrix: A scipy.csr_matrix
    """

    cv = TfidfVectorizer(stop_words='english',
                         ngram_range=(1, 2),
                         lowercase=True,
                         min_df=2,
                         max_features=20000)

    train_matrix = cv.fit_transform(train_data)
    test_matrix = cv.transform(test_data)

    return cv, train_matrix, test_matrix


def apply_nmf(data, random_state):
    """
    Applies non-negative matrix factorization (NMF) to compute topics.

    Parameters
    ----------
    data: A csr_matrix
    random_state: A RandomState instance for NMF

    Returns
    -------
    A tuple of (nmf, transformed_data)
    nmf: An sklearn.NMF instance
    transformed_data: A numpy.ndarray
    """

    nmf = NMF(n_components=60, max_iter=200, random_state=random_state).fit(data)
    td = nmf.transform(data)
    td_norm = normalize(td, norm='l1', axis=1)

    return nmf, td_norm


def classify_topics(nmf, X_train, y_train, X_test, random_state):
    """

    Parameters
    ---------
    nmf: An sklearn.NMF model.
    X_train: A numpy array.
    y_train: A numpy array.
    X_test: A scipy csr_matrix.
    random_state: A RandomState instance for Random Forest Classifier.

    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A RandomForestClassifier instance.
    y_pred: A numpy array.
    """

    clf = RandomForestClassifier(random_state=random_state).fit(X_train, y_train)
    y_pred = clf.predict(nmf.transform(X_test))

    return clf, y_pred


def get_topics(cv, train_data):
    """
    Uses gensim to perform topic modeling.

    Parameters
    ---------
    cv: A TfidfVectorizer instance.
    train_data: A scipy csr_matrix.

    Returns
    -------
    A list of strings (functions of the most important terms in each topic).
    """

    td_gensim = Sparse2Corpus(train_data, documents_columns=False)
    tmp_dct = dict((idv, word) for word, idv in cv.vocabulary_.items())
    dct = Dictionary.from_corpus(td_gensim, id2word=tmp_dct)

    lda = LdaModel(corpus=td_gensim, id2word=dct, num_topics=20)
    topics = lda.top_topics(corpus=td_gensim, num_words=5)

    return topics


def find_number_of_entries_in_synonym_ring(word):
    """
    Finds the number of entries in the wordnet synset.

    Parameters
    ----------
    word: A string.

    Returns
    -------
    An int.
    """

    the_synsets = wn.synsets(word)

    return len(the_synsets)


def get_path_similarity_between_boy_and_girl():
    """
    Computes the path similarity between "boy" and "girl".

    Returns
    -------
    A float.
    """

    return wn.path_similarity(wn.synset('boy.n.01'), wn.synset('girl.n.01'))


def get_path_similarity_between_boy_and_cat():
    """
    Computes the path similarity between "boy" and "cat".

    Returns
    -------
    A float.
    """

    return wn.path_similarity(wn.synset('boy.n.01'), wn.synset('cat.n.01'))


def get_path_similarity_between_boy_and_dog():
    """
    Computes the path similarity between "boy" and "dog".

    Returns
    -------
    A float.
    """

    return wn.path_similarity(wn.synset('boy.n.01'), wn.synset('dog.n.01'))


def get_path_similarity_between_girl_and_girl():
    """
    Computes the path similarity between "girl" and "girl".

    Returns
    -------
    A float.
    """

    # YOUR CODE HERE

    return wn.path_similarity(wn.synset('girl.n.01'), wn.synset('girl.n.01'))


def get_model(sentences):
    """
    Builds a Word2Vec model from "corpus"

    Parameters
    ----------
    sentences: An NTLK corpus.

    Returns
    -------
    A Word2Vec instance.
    """

    return gensim.models.Word2Vec(sentences, window=10, min_count=2)


def get_cosine_similarity(model, word1, word2):
    """
    Computes cosine similarity between "word1" and "word2" using a Word2Vec model.

    Parameters
    ----------
    model: A gensim.Word2Vec model.
    word1: a word
    word2: another word

    Returns
    -------
    A float

    """

    return model.similarity(word1, word2)


def find_most_similar_words(model):
    """
    Find the top 3 most similar words,
    where "girl" and "cat" contribute positively towards the similarity,
    and "boy" and "dog" contribute negatively.

    Parameters
    ----------
    model: A gensim.Word2Vec model.

    Returns
    -------
    A list of tuples (word, similarty).
    word: A string.
    similarity: A float.
    """

    result = list()
    words = [t[0] for t in model.most_similar(positive=['girl', 'cat'], negative=['boy', 'dog'], topn=3)]
    similarities = [t[1] for t in model.most_similar(positive=['girl', 'cat'], negative=['boy', 'dog'], topn=3)]

    for i in range(3):
        result.append((words[i], similarities[i]))

    return result
