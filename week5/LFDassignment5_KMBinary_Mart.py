import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score, v_measure_score, homogeneity_completeness_v_measure, confusion_matrix, homogeneity_score, completeness_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from collections import defaultdict, Counter


def read_corpus(corpus_file, use_sentiment):
	"Reads in the corpus and returns the documents and labels"
	documents = []
	labels = []
	snowballstemmer = SnowballStemmer('english', ignore_stopwords = False)
	sentimentClasses = ['health', 'software']
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens = line.strip().split()
			use_stopword = False
			use_grams = False
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
			elif use_grams:
				stemmedtokens = [snowballstemmer.stem(word) for word in tokens[3:]]
				documents.append(find_ngrams(stemmedtokens, 2))
			else:
				stemmedtokens = [snowballstemmer.stem(word) for word in tokens[3:]]
				if use_sentiment:
					documents.append(stemmedtokens)
				else:
					if tokens[0] in sentimentClasses:
						documents.append(stemmedtokens)

			if use_sentiment:
				# 2-class problem: positive vs negative
				labels.append( tokens[1] )
			else:
				# 6-class problem: books, camera, dvd, health, music, software
				if tokens[0] in sentimentClasses:
					labels.append( tokens[0] )

	return documents, labels


def find_ngrams(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

def identity(x):
	return x

def bestClassify(X,Y):
	"Best classifier function"
	tfidf = True

	if tfidf:
		vec = TfidfVectorizer(preprocessor = identity,
							tokenizer = identity, sublinear_tf = True)
	else:
		vec = CountVectorizer(preprocessor = identity,
							tokenizer = identity)

	km = KMeans(n_clusters=2, n_init=100, verbose=1)
	clusterer = Pipeline( [('vec', vec),
								('cls', km)] )

	prediction = clusterer.fit_predict(X,Y)

	checker = defaultdict(list)
	for pred,truth in zip(prediction,Y):
		checker[pred].append(truth)

	labeldict = {}
	for pred, label in checker.items():
		labeldict[pred] = Counter(label).most_common(1)[0][0]
		#print(pred, Counter(label).most_common(1)[0][0])

	prediction = [labeldict[p] for p in prediction]
	labels = list(labeldict.values())
	print(labels)
	print(confusion_matrix(Y, prediction, labels=labels))

	print("Homogeneity:", homogeneity_score(Y,prediction))
	print("Completeness:", completeness_score(Y,prediction))
	print("V-measure:", v_measure_score(Y,prediction))
	print("Rand-Index:", adjusted_rand_score(Y,prediction))


X, Y = read_corpus('all_sentiment_shuffled.txt', False)
bestClassify(X,Y)

"""
if len(sys.argv) != 3:
	print("Usage: python LFDassignment5_SVM_Mart.py <trainset> <testset>", file=sys.stderr)
	exit(-1)
Xtrain, Ytrain = read_corpus(sys.argv[1], True)
Xtest, Ytest = read_corpus(sys.argv[2], True)
Yguess = bestClassify(Xtrain,Ytrain, Xtest, Ytest)
print(classification_report(Ytest, Yguess))"""

