#!usr/bin/python3.4

import sys, glob, nltk, re, string
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from nltk.stem import WordNetLemmatizer


def fileRead(language, setting):
	""" Reads in data from the directories """
	informationDict = defaultdict(list)
	documents = []
	genders = []
	ages = []
	authors = []
	if setting == 'training':
		truthFile = ''.join([truthfile for truthfile in glob.glob(setting + '/' + language + '/truth.txt')])
		for line in open(truthFile, 'r'):
			truthvalues = [value for value in line.strip().split(':::')]
			informationDict[truthvalues[0]] = (truthvalues[1], truthvalues[2])
	fileList = [xmlfile for xmlfile in glob.glob(setting + '/' + language + '/*.xml')]
	for xmlfile in fileList:
		tree = ET.parse(xmlfile)
		root = tree.getroot()
		for child in root:
			tweet = child.text.strip()
			documents.append(child.text.strip())
			if setting == 'training':
				genders.append(informationDict[root.attrib['id']][0])
				ages.append(informationDict[root.attrib['id']][1])
			authors.append(root.attrib['id'])
	if setting == 'training':
		return documents, genders, ages, authors
	else:
		return documents, authors

def identity(x):
	""" Placeholder function for preprocessing and tokenizing function, for easily switching """
	return x

def tokenizerStemmer(document):
	""" Tokenizer/Lemmatizer function """
	words = document.split()
	tokens = []
	wnl = WordNetLemmatizer()
	for word in words:
		tokens.append(wnl.lemmatize(word))
	return ' '.join(tokens)
	#return ' '.join(nltk.word_tokenize(document))

def preprocessing(document):
	""" Small preprocessing function which, after testing different settings, became fully obsolete """
	exclude = set(string.punctuation)
	document = document.lower()
	document = re.sub(r"http\S+", "link", document)
	document = ''.join(ch for ch in document if ch not in exclude)
	return document

def trainGenderclassification(doc, gen, lang):
	""" A function for testing out different settings to increase scores and present final train scores """
	splitPoint = int(0.8*len(doc))
	xTrain = doc[:splitPoint]
	xTest = doc[splitPoint:]
	yTrain = gen[:splitPoint]
	yTest = gen[splitPoint:]


	unionOfFeatures = FeatureUnion([
									('normaltfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
									('bigrams', TfidfVectorizer(preprocessor = identity, tokenizer = identity, ngram_range = (2,2), analyzer = 'char')),
									('counts', CountVectorizer(preprocessor = identity, tokenizer = identity))
									])

	#classifier = Pipeline([('vec', TfidfVectorizer()), ('cls', svm.SVC(kernel='rbf', C=1.5))])
	featureFit = unionOfFeatures.fit(xTrain, yTrain).transform(xTrain)
	classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.5))])
	classifier.fit(xTrain, yTrain)

	yGuess = classifier.predict(xTest)
	print(classification_report(yTest, yGuess))
	print(lang, "gender accuracy score:")
	print(accuracy_score(yTest, yGuess))
	

def trainAgeclassification(doc, age, lang):
	""" A function for testing out different settings to increase scores """
	splitPoint = int(0.8*len(doc))
	xTrain = doc[:splitPoint]
	xTest = doc[splitPoint:]
	yTrain = age[:splitPoint]
	yTest = age[splitPoint:]

	unionOfFeatures = FeatureUnion([
									('normaltfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
									('bigrams', TfidfVectorizer(preprocessor = identity, tokenizer = identity, ngram_range = (3,3), analyzer = 'char')),
									('counts', CountVectorizer(preprocessor = identity, tokenizer = identity))
									])

	
	featureFit = unionOfFeatures.fit(xTrain, yTrain).transform(xTrain)
	classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.5))])
	classifier.fit(xTrain, yTrain)

	yGuess = classifier.predict(xTest)
	print(classification_report(yTest, yGuess))
	print(lang, "age accuracy score:")
	print(accuracy_score(yTest, yGuess))

def genderClassifier(doc, gen):
	""" A function that trains a gender classifier """
	xTrain = doc
	yTrain = gen

	unionOfFeatures = FeatureUnion([
									('normaltfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
									('bigrams', TfidfVectorizer(preprocessor = identity, tokenizer = identity, ngram_range = (2,2), analyzer = 'char')),
									('counts', CountVectorizer(preprocessor = identity, tokenizer = identity))
									])

	featureFit = unionOfFeatures.fit(xTrain, yTrain).transform(xTrain)
	classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.5))])
	classifier.fit(xTrain, yTrain)
	
	return classifier

def ageClassifier(doc, age):
	""" A function that trains an age classifier """
	xTrain = doc
	yTrain = age

	unionOfFeatures = FeatureUnion([
									('normaltfidf', TfidfVectorizer(preprocessor = identity, tokenizer = identity)),
									('bigrams', TfidfVectorizer(preprocessor = identity, tokenizer = identity, ngram_range = (3,3), analyzer = 'char')),
									('counts', CountVectorizer(preprocessor = identity, tokenizer = identity))
									])

	featureFit = unionOfFeatures.fit(xTrain, yTrain).transform(xTrain)
	classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.5))])
	classifier.fit(xTrain, yTrain)
	
	return classifier

def makePredictions(genderclassifier, ageclassifier, testdocs, testauthors):
	""" A function that predicts the genders and age categories for test data """
	combinationList = []
	for document, author in zip(testauthors, testdocs):
		combinationList.append([author, document])

	for pairlist in combinationList:
		pairlist.append(genderclassifier.predict([pairlist[0]]))
		if ageclassifier == 'dutchAgeClassifier':
			pairlist.append('XX-XX')
		elif ageclassifier == 'italianAgeClassifier':
			pairlist.append('XX-XX')
		else:
			pairlist.append(ageclassifier.predict([pairlist[0]]))

	return combinationList

def writeTruthFile(language, combinationList):
	""" A function that writes the predictions to the corrosponding truth files """
	filename = 'test/' + language + '/truth.txt'
	truthfile = open(filename, 'w')
	genderDict = defaultdict(list)
	ageDict = defaultdict(list)
	if language == 'english' or language == 'spanish':
		for combination in combinationList:
			genderDict[combination[1]].append(combination[2][0])
			ageDict[combination[1]].append(combination[3][0])
		for key, value in genderDict.items():
			truthfile.write(str(key) + ':::' + str(Counter(value).most_common(1)[0][0]) + ':::' + str(Counter(ageDict[key]).most_common(1)[0][0]) + '\n')
	else:
		for combination in combinationList:
			genderDict[combination[1]].append(combination[2][0])
		for key, value in genderDict.items():
			truthfile.write(str(key) + ':::' + str(Counter(value).most_common(1)[0][0]) + ':::' + 'XX-XX' + '\n')

if __name__ == '__main__':
	"""# For training
	language = 'english'
	if language == 'english':
		engDOC, engGEN, engAGE, engAUT = fileRead(language, 'training')
		trainGenderclassification(engDOC, engGEN, 'English')
		trainAgeclassification(engDOC, engAGE, 'English')
	language = 'dutch'
	if language == 'dutch':
		dutDOC, dutGEN, dutAGE, dutAUT = fileRead(language, 'training')
		trainGenderclassification(dutDOC, dutGEN, 'Dutch')
		#trainAgeclassification(dutDOC, dutAGE)
	language = 'italian'
	if language == 'italian':
		itaDOC, itaGEN, itaAGE, itaAUT = fileRead(language, 'training')
		trainGenderclassification(itaDOC, itaGEN, 'Italian')
		#trainAgeclassification(itaDOC, itaAGE)
	language = 'spanish'
	if language == 'spanish':
		spaDOC, spaGEN, spaAGE, spaAUT = fileRead(language, 'training')
		trainGenderclassification(spaDOC, spaGEN, 'Spanish')
		trainAgeclassification(spaDOC, spaAGE, 'Spanish')"""
	
	
	
	print("Reading in files...", file=sys.stderr)
	trainengDOC, trainengGEN, trainengAGE, trainengAUT = fileRead('english', str(sys.argv[1]))
	testengDOC, testengAUT = fileRead('english', str(sys.argv[2]))
	traindutDOC, traindutGEN, traindutAGE, traindutAUT = fileRead('dutch', str(sys.argv[1]))
	testdutDOC, testdutAUT = fileRead('dutch', str(sys.argv[2]))
	trainitaDOC, trainitaGEN, trainitaAGE, trainitaAUT = fileRead('italian', str(sys.argv[1]))
	testitaDOC, testitaAUT = fileRead('italian', str(sys.argv[2]))
	trainspaDOC, trainspaGEN, trainspaAGE, trainspaAUT = fileRead('spanish', str(sys.argv[1]))
	testspaDOC, testspaAUT = fileRead('spanish', str(sys.argv[2]))
	print("Reading in files: Done!", file=sys.stderr)
	
	print("Training classifiers...", file=sys.stderr)
	englishGenderClassifier = genderClassifier(trainengDOC, trainengGEN)
	englishAgeClassifier = ageClassifier(trainengDOC, trainengAGE)
	dutchGenderClassifier = genderClassifier(traindutDOC, traindutGEN)
	#dutchAgeClassifier = ageClassifier(traindutDOC, traindutAGE)
	italianGenderClassifier = genderClassifier(trainitaDOC, trainitaGEN)
	#italianAgeClassifier = ageClassifier(trainitaDOC, trainitaAGE)
	spanishGenderClassifier = genderClassifier(trainspaDOC, trainspaGEN)
	spanishAgeClassifier = ageClassifier(trainspaDOC, trainspaAGE)
	print("Training classifiers: Done!", file=sys.stderr)

	print("Making predictions...", file=sys.stderr)
	englishPredictions = makePredictions(englishGenderClassifier, englishAgeClassifier, testengDOC, testengAUT)
	dutchPredictions = makePredictions(dutchGenderClassifier, 'dutchAgeClassifier', testdutDOC, testdutAUT)
	italianPredictions = makePredictions(italianGenderClassifier, 'italianAgeClassifier', testitaDOC, testitaAUT)
	spanishPredictions = makePredictions(spanishGenderClassifier, spanishAgeClassifier, testspaDOC, testspaAUT)
	print("Making predictions: Done!", file=sys.stderr)

	print("Writing truth files...", file=sys.stderr)
	writeTruthFile('english', englishPredictions)
	writeTruthFile('dutch', dutchPredictions)
	writeTruthFile('italian', italianPredictions)
	writeTruthFile('spanish', spanishPredictions)
	print("Writing truth files: Done!", file=sys.stderr)



