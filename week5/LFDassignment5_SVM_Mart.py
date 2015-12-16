import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from nltk.stem.snowball import SnowballStemmer


def read_corpus(corpus_file, use_sentiment):
	"Reads in the corpus and returns the documents and labels"
	documents = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens = line.strip().split()
			use_stopword = False
			if use_stopword:
				stopwordfile = open('stopwords.txt', 'r')
				stopwords = []
				for line in stopwordfile:
					if len(line) > 0:
						splitline = line.split(',')
						for word in splitline:
							stopwords.append(word)

				tokenlist = [token for token in tokens[3:] if token not in stopwords]
				documents.append(find_ngrams(tokenlist, 2))
			else:
				snowballstemmer = SnowballStemmer('english')
				stemmedtokens = [snowballstemmer.stem(word) for word in tokens[3:]]
				#documents.append(stemmedtokens)
				documents.append(find_ngrams(stemmedtokens, 2))
			if use_sentiment:
				# 2-class problem: positive vs negative
				labels.append( tokens[1] )
			else:
				# 6-class problem: books, camera, dvd, health, music, software
				labels.append( tokens[0] )

	return documents, labels


def find_ngrams(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

def identity(x):
	return x

def crosvalClassifier(X,Y):
	"A classifier that uses cross-validation for training purposes"

	tfidf = True

	if tfidf:
		vec = TfidfVectorizer(preprocessor = identity,
							tokenizer = identity, stop_words = None)
	else:
		vec = CountVectorizer(preprocessor = identity,
							tokenizer = identity)

	C = 1.0
	cls = svm.SVC(kernel='linear', C=C)
	classifier = Pipeline( [('vec', vec),
								('cls', cls)] )
	print("All OK so far")
	#crossvalidator = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="accuracy")
	crossvalidatorPrecision = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="precision_weighted")
	crossvalidatorRecall = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="recall_weighted")
	crossvalidatorFscore = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="f1_weighted")

	#print("Accuracy:", sum(crossvalidator)/len(crossvalidator))
	print("Precision:", sum(crossvalidatorPrecision)/len(crossvalidatorPrecision))
	print("Recall:", sum(crossvalidatorRecall)/len(crossvalidatorRecall))
	print("F1-score:", sum(crossvalidatorFscore)/len(crossvalidatorFscore))
	return sum(crossvalidatorFscore/len(crossvalidatorFscore))

def bestClassify(Xtrain,Ytrain, Xtest, Ytest):
	"Best classifier function"
	tfidf = True

	if tfidf:
		vec = TfidfVectorizer(preprocessor = identity,
							tokenizer = identity, )
	else:
		vec = CountVectorizer(preprocessor = identity,
							tokenizer = identity)

	C = 1.0
	cls = svm.SVC(kernel='linear', C=C)
	classifier = Pipeline( [('vec', vec),
								('cls', cls)] )

	classifier.fit(Xtrain, Ytrain)
	Yguess = classifier.predict(Xtest)
	
	return Yguess

"""
X, Y = read_corpus('trainset.txt', True)
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

#Yguess = bestClassify(Xtrain, Ytrain, Xtest, Ytest)
#print(classification_report(Ytest, Yguess))
#crosvalClassifier(X,Y)"""


if len(sys.argv) != 3:
	print("Usage: python LFDassignment5_SVM_Mart.py <trainset> <testset>", file=sys.stderr)
	exit(-1)
Xtrain, Ytrain = read_corpus(sys.argv[1], True)
Xtest, Ytest = read_corpus(sys.argv[2], True)
Yguess = bestClassify(Xtrain,Ytrain, Xtest, Ytest)
print(classification_report(Ytest, Yguess))

