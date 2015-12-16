from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import sys
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# This function reads a textfile, splits the document on spaces into a list and depending on the Boolean
# parameter use_sentiment which labels are used. the Document list contains all tokens and the labels the correct labels depending
# on the use_sentiment Boolean. This functions returns two lists, one with lists of lists with tokens of each document and the list variable
# returns all possible labels
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    bigrams = True
    #stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    wordnet_lemmatizer = WordNetLemmatizer()

    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            t = [stemmer.stem(token) for token in tokens[3:]]
            #t = [wordnet_lemmatizer.lemmatize(token) for token in tokens[3:]]
            
            if bigrams:   
                documents.append(list(zip(t, t[1:])))
                #print(list(zip(t, t[1:])))
            else:
                documents.append(t)
            labels.append( tokens[1] )
    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x



def main(argv):
    testmode = False #seperate testfile or do cross validation

    if len(argv) == 2:
        trainfile = argv[1]
    elif len(argv) == 3:
        testmode = True
        trainfile = argv[1]
        testfile = argv[2]
    else:
        exit("Use assignment.py trainfile <testfile> <Optional>")
    

    # X and Y are the result of the read corpus function. X is a list of all documents that are tokenized and Y is a list of all labels
    # The use_sentiment boolean can be changed to use the categories(False) or the polarity(True)
    X, Y = read_corpus(trainfile, use_sentiment=True)
    print(X)
    if testmode:
        print("Use test file")
        Xtrain = X
        Ytrain = Y
        Xtest, Ytest = read_corpus(testfile, use_sentiment=True)
    else:
        #this code splits the data in a training and test set (75% train and 25% test)
        print("Use 75\% of train file")
        split_point = int(0.75*len(X))
        Xtrain = X[:split_point]
        Ytrain = Y[:split_point]
        Xtest = X[split_point:]
        Ytest = Y[split_point:]

    tfidf = True
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity,
                              tokenizer = identity, sublinear_tf=True, max_df=0.05)
    else:
        vec = CountVectorizer(preprocessor = identity,
                              tokenizer = identity)
    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.SVC(kernel='rbf', C=1.0, gamma=0.9))] )

    # Train the classifier using 75% of the data
    classifier.fit(Xtrain, Ytrain)

    # Use the classifier for the remaining 25% of the data
    Yguess = classifier.predict(Xtest)

    #possible labels
    labels = list(set(Ytest))
    print(classification_report(Ytest, Yguess))

    


main(sys.argv)