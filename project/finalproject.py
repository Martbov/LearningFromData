#!usr/bin/python3.4

import sys
import glob
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import cross_val_score
import nltk
from sklearn import svm

def fileRead(language, setting):
	"""Reads in data from the directories"""
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
	#for author in fileList:
	#	print(author)
	#	print(author.split('/')[2].split('.')[0])
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
	return x

def tokenizer(document):
	"""Tokenizer function"""
	return ' '.join(nltk.word_tokenize(document))

def preprocessing(document):
	"""Small preprocessing function"""
	oldWords = document.split()
	# Attempt to exclude punctuation from the tweets, but results were lower than tweets with punctuation
	"""punctList = ['!', '?', '@' ',', '.', '(', ')', '/', '\\', '|', '“', '"']
	for i in range(len(oldWords)):
		for char in oldWords[i]:
			if char in punctList:
				oldWords[i] = oldWords[i].replace(char, '')"""
	newWords = []
	
	for word in oldWords:
		if word.startswith('#'):
			newWords.append('hashtag')
		elif word.startswith('http'):
			newWords.append('LINK')
		else:
			newWords.append(word.lower())
	#taggedWords = nltk.pos_tag(newWords)
	#print(taggedWords)
	#print(' '.join(newWords))
	return ' '.join(newWords)


def genderClassfier(doc, gen):
	"""A classifier for classifying gender"""
	splitPoint = int(0.8*len(doc))
	xTrain = doc
	#xTest = doc[splitPoint:]
	yTrain = gen
	#yTest = gen[splitPoint:]

	#vec = TfidfVectorizer(preprocessor = preprocessing, sublinear_tf = True)
	unionOfFeatures = FeatureUnion([
									('normaltfidf', TfidfVectorizer(preprocessor = preprocessing, max_features = 800)),
									('bigrams', TfidfVectorizer(preprocessor = preprocessing, ngram_range = (2,2), analyzer = 'char', max_features = 800)),
									('counts', CountVectorizer(preprocessor = preprocessing, max_features = 800))
									])

	featureFit = unionOfFeatures.fit(xTrain, yTrain).transform(xTrain)
	classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.5))])
	classifier.fit(xTrain, yTrain)
	#yGuess = classifier.predict(xTest)
	#print(classification_report(yTest, yGuess))
	
	return classifier

def ageClassifier(doc, age):
	"""A classifier for classifying age"""
	splitPoint = int(0.8*len(doc))
	xTrain = doc
	#xTest = doc[splitPoint:]
	yTrain = age
	#yTest = age[splitPoint:]

	#vec = TfidfVectorizer(preprocessor = preprocessing, sublinear_tf = True)
	unionOfFeatures = FeatureUnion([
									('normaltfidf', TfidfVectorizer(preprocessor = preprocessing, max_features = 800)),
									('bigrams', TfidfVectorizer(preprocessor = preprocessing, ngram_range = (2,2), analyzer = 'char', max_features = 800)),
									('counts', CountVectorizer(preprocessor = preprocessing, max_features = 800))
									])

	featureFit = unionOfFeatures.fit(xTrain, yTrain).transform(xTrain)
	classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.5))])
	classifier.fit(xTrain, yTrain)
	#yGuess = classifier.predict(xTest)
	#print(classification_report(yTest, yGuess))
	
	return classifier

def makePredictions(genderclassifier, ageclassifier, testdocs, testauthors):
	combinationList = []
	for document, author in zip(testauthors, testdocs):
		combinationList.append([author, document])

	for pairlist in combinationList:
		pairlist.append(genderclassifier.predict(pairlist[1]))
		if ageclassifier == 'dutchAgeClassifier':
			pairlist.append('XX-XX')
		elif ageclassifier == 'italianAgeClassifier':
			pairlist.append('XX-XX')
		else:
			pairlist.append(ageclassifier.predict(pairlist[1]))

	return combinationList

def writeTruthFile(language, combinationList):
	filename = 'truth' + language.upper()[:3] + '.txt'
	if language == 'english':
		for combination in combinationList:
			print(combination[1], Counter(list(combination[2])).most_common(1)[0], Counter(list(combination[3])).most_common(1)[0])
		

if __name__ == '__main__':
	"""# For training
	if len(sys.argv) != 2:
		print("Usage: python finalproject.py <language>", file=sys.stderr)
	else:
		language = sys.argv[1]
		if language == 'english':
			engDOC, engGEN, engAGE, engAUT = fileRead(language, 'training')
			#genderClassfier(engDOC, engGEN)
			#ageClassifier(engDOC, engAGE)
		elif language == 'dutch':
			dutDOC, dutGEN, dutAGE, dutAUT = fileRead(language, 'training')
			genderClassfier(dutDOC, dutGEN)
			ageClassifier(dutDOC, dutAGE)
		elif language == 'italian':
			itaDOC, itaGEN, itaAGE, itaAUT = fileRead(language, 'training')
			genderClassfier(itaDOC, itaGEN)
			ageClassifier(itaDOC, itaAGE)
		elif language == 'spanish':
			spaDOC, spaGEN, spaAGE, spaAUT = fileRead(language, 'training')
			genderClassfier(spaDOC, spaGEN)
			ageClassifier(spaDOC, spaAGE)
		else:
			print("Use language 'english', 'dutch', 'spanish' or 'italian'")"""

	print("Reading in files...", file=sys.stderr)
	trainengDOC, trainengGEN, trainengAGE, trainengAUT = fileRead('english', 'training')
	testengDOC, testengAUT = fileRead('english', 'testing')
	traindutDOC, traindutGEN, traindutAGE, traindutAUT = fileRead('dutch', 'training')
	testdutDOC, testdutAUT = fileRead('dutch', 'testing')
	trainitaDOC, trainitaGEN, trainitaAGE, trainitaAUT = fileRead('italian', 'training')
	testitaDOC, testitaAUT = fileRead('italian', 'testing')
	trainspaDOC, trainspaGEN, trainspaAGE, trainspaAUT = fileRead('spanish', 'training')
	testspaDOC, testspaAUT = fileRead('spanish', 'testing')
	print("Reading in files: Done!", file=sys.stderr)

	print("Training classifiers...", file=sys.stderr)
	englishGenderClassifier = genderClassfier(trainengDOC, trainengGEN)
	englishAgeClassifier = ageClassifier(trainengDOC, trainengAGE)
	#dutchGenderClassifier = genderClassfier(traindutDOC, traindutGEN)
	#dutchAgeClassifier = ageClassifier(traindutDOC, traindutAGE)
	#italianGenderClassifier = genderClassfier(trainitaDOC, trainitaGEN)
	#italianAgeClassifier = ageClassifier(trainitaDOC, trainitaAGE)
	#spanishGenderClassifier = genderClassfier(trainspaDOC, trainspaGEN)
	#spanishAgeClassifier = ageClassifier(trainspaDOC, trainspaAGE)
	print("Training classifiers: Done!", file=sys.stderr)

	print("Making predictions...", file=sys.stderr)
	englishPredictions = makePredictions(englishGenderClassifier, englishAgeClassifier, testengDOC, testengAUT)
	#dutchPredictions = makePredictions(dutchGenderClassifier, 'dutchAgeClassifier', testdutDOC, testdutAUT)
	#italianPredictions = makePredictions(italianGenderClassifier, 'italianAgeClassifier', testitaDOC, testitaAUT)
	#spanishPredictions = makePredictions(spanishGenderClassifier, spanishAgeClassifier, testspaDOC, testspaAUT)
	print("Making predictions: Done!", file=sys.stderr)

	print("Writing truth files...", file=sys.stderr)
	writeTruthFile('english', englishPredictions)
	#writeTruthFile('dutch', dutchPredictions)
	#writeTruthFile('italian', italianPredictions)
	#writeTruthFile('spanish', spanishPredictions)
	print("Writing truth files: Done!", file=sys.stderr)



