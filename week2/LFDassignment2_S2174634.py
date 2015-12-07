import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from matplotlib import pyplot as plt
from progressbar import ProgressBar
import time

def read_corpus(corpus_file):
	"Function that reads the corpus given as an argument and based on the second argument uses the the binary or multinomial classification."
	documents = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens = line.strip().split()

			documents.append(tokens[3:])
			labels.append( tokens[0] )

	return documents, labels

def neighborClassification(k):
	"kNN classifier function"
	classifier = Pipeline( [('vec', vec),
							('cls', KNeighborsClassifier(n_neighbors=k))] )

	crossvalidator = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1)
	#crossvalidatorPrecision = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="precision_weighted")
	#crossvalidatorRecall = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="recall_weighted")
	crossvalidatorFscore = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="f1_weighted")

	#print("Accuracy:", sum(crossvalidator)/len(crossvalidator))
	#print("Precision:", sum(crossvalidatorPrecision)/len(crossvalidatorPrecision))
	#print("Recall:", sum(crossvalidatorRecall)/len(crossvalidatorRecall))
	#print("F1-score:", sum(crossvalidatorFscore)/len(crossvalidatorFscore))
	return sum(crossvalidatorFscore/len(crossvalidatorFscore))


def plotgraph():
	"Function to plot the graph for kNN classifier"
	kList = []
	scoreList = []
	pbar = ProgressBar()
	for i in pbar(range(1,20)):
		kList.append(i)
		scoreList.append(neighborClassification(i))
		print("k =", i, neighborClassification(i))
		
	plt.plot(kList, scoreList)

	plt.title('K Nearest Neighbors')
	plt.ylabel('F-Score')
	plt.xlabel('K')
	plt.show()

def identity(x):
	return x

def crosvalClassifier(X,Y):
	"A classifier that uses cross-validation"

	tfidf = True

	if tfidf:
		vec = TfidfVectorizer(preprocessor = identity,
							tokenizer = identity)
	else:
		vec = CountVectorizer(preprocessor = identity,
							tokenizer = identity)

	classifier = Pipeline( [('vec', vec),
							('cls', MultinomialNB(alpha=0.23))] )

	crossvalidator = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1)
	#crossvalidatorPrecision = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="precision_weighted")
	#crossvalidatorRecall = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="recall_weighted")
	crossvalidatorFscore = cross_val_score(classifier, X, y=Y, cv=5, n_jobs=-1, scoring="f1_weighted")

	#print("Accuracy:", sum(crossvalidator)/len(crossvalidator))
	#print("Precision:", sum(crossvalidatorPrecision)/len(crossvalidatorPrecision))
	#print("Recall:", sum(crossvalidatorRecall)/len(crossvalidatorRecall))
	#print("F1-score:", sum(crossvalidatorFscore)/len(crossvalidatorFscore))
	return sum(crossvalidatorFscore/len(crossvalidatorFscore))

def bestClassify(Xtrain,Ytrain, Xtest, Ytest):
	"Best classifier function"
	tfidf = True

	if tfidf:
		vec = TfidfVectorizer(preprocessor = identity,
							tokenizer = identity)
	else:
		vec = CountVectorizer(preprocessor = identity,
							tokenizer = identity)


	classifier = Pipeline( [('vec', vec),
								('cls', MultinomialNB(alpha=0.23))] )

	t0 = time.time()
	classifier.fit(Xtrain, Ytrain)
	train_time = time.time() - t0
	t1 = time.time()
	classifier.predict(Xtest)
	Yguess = classifier.predict(Xtest)
	test_time = time.time() - t1

	#print("Train time:", train_time)
	#print("Test time", test_time)

	return Yguess

if len(sys.argv) != 3:
	print("Usage: python LFDassignment2_S2174634.py trainset.txt testset.txt", file=sys.stderr)
	exit(-1)

# X and Y are the returned documents and labels respectively. X and Y are both splitted at 75%, which implies 75% of the data is used as traindata and the rest as testdata.
Xtrain, Ytrain = read_corpus(sys.argv[1])
Xtest, Ytest = read_corpus(sys.argv[2])

Yguess = bestClassify(Xtrain,Ytrain, Xtest, Ytest)
print(classification_report(Ytest, Yguess))

