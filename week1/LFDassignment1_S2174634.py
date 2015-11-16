import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

# Function that reads the corpus given as an argument and based on the second argument uses the the binary or multinomial classification.
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

# X and Y are the returned documents and labels respectively. X and Y are both splitted at 75%, which implies 75% of the data is used as traindata and the rest as testdata.
sentiment = sys.argv[1]
if sentiment == "Multinomial":
    sentiment = False
elif sentiment == "Binary":
    sentiment = True
else:
    print("Usage: python LFDassignment1.py Multinomial or python LFDassignment1 Binary", file=sys.stderr)
    exit(-1)

X, Y = read_corpus('all_sentiment_shuffled.txt', sentiment)
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )


# Trains the classifier based on the training documents and labels.
classifier.fit(Xtrain, Ytrain)

# Using the trained classifier to predict the labels for the testdata.
Yguess = classifier.predict(Xtest)

# Prints the accuracy scores, which compares the correct labels from Ytest with predicted ones from the system in Yguess.
#print(classification_report(Ytest, Yguess))
#print("Overal accuracy:", accuracy_score(Ytest, Yguess))
#print(classifier.predict_proba(Xtest))

labelbin = LabelBinarizer()
print(labelbin.fit_transform(classes))



