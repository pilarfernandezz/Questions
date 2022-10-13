import nltk
import sys
import os
import string
import math
FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictMap = dict()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            dictMap[file] = f.read()
    return dictMap


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    wordList = []
    for word in nltk.word_tokenize(document.lower()):
        if word not in nltk.corpus.stopwords.words("english") and word not in string.punctuation:
            wordList.append(word)
    return wordList


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfDict = dict()
    for document in documents:
        for word in documents[document]:
            numberOfOcc = 0
            for doc in documents:
                if word in documents[doc]:
                    numberOfOcc += 1
            idfDict[word] =  math.log(len(documents) / numberOfOcc)
    return idfDict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    topFiles = dict()
    for file in files: 
        tfidf = 0
        for word in query:
            tfidf += files[file].count(word)  * idfs[word]
        topFiles[file] = tfidf
    topFiles = sorted(topFiles, key=lambda item: topFiles[item], reverse=True)
    
    nTopFiles = list(topFiles)[0: n]
    return nTopFiles


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    topSentences = dict()
    for sentence in sentences:
        matchingWordMeasure, queryTermDensity = 0, 0
        words = sentences[sentence]
        for word in query:
            if word in words:
                queryTermDensity += words.count(word) / len(words)  
                matchingWordMeasure += idfs[word]
        topSentences[sentence] = [matchingWordMeasure, queryTermDensity]

    topSentences = sorted(topSentences, key=lambda item: topSentences[item], reverse=True)   

    nTopSentences = list(topSentences)[0: n]
    return nTopSentences


if __name__ == "__main__":
    main()
