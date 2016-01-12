#!usr/bin/python3.4

import sys
import glob
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import cross_val_score
import nltk

def fileRead(language):
	"""Reads in data from the directories"""
	informationDict = defaultdict(list)
	documents = []
	genders = []
	ages = []
	authors = []
	truthFile = ''.join([truthfile for truthfile in glob.glob('training/' + language + '/truth.txt')])
	for line in open(truthFile, 'r'):
		truthvalues = [value for value in line.strip().split(':::')]
		informationDict[truthvalues[0]] = (truthvalues[1], truthvalues[2])
	fileList = [xmlfile for xmlfile in glob.glob('training/' + language + '/*.xml')]
	for xmlfile in fileList:
		tree = ET.parse(xmlfile)
		root = tree.getroot()
		for child in root:
			tweet = child.text.strip()
			documents.append(child.text.strip())
			genders.append(informationDict[root.attrib['id']][0])
			ages.append(informationDict[root.attrib['id']][1])
			authors.append(root.attrib['id'])
	return documents, genders, ages, authors

def tokenizer(document):
	"""Tokenizer function"""
	return ' '.join(nltk.word_tokenize(document))

def preprocessing(document):
	"""Small preprocessing function"""
	oldWords = document.split()
	newWords = []
	punctList = ['!', '?', '@' ',', '.', '(', ')', '#', '/', '\\', '|']
	for word in oldWords:
		if word.startswith('#'):
			newWords.append('#hashtag')
		elif word.startswith('http'):
			newWords.append('LINK')
		else:
			newWords.append(word)
	return ' '.join(newWords)


def naiveBayesClassfier(doc, gen, age, aut):
	"""A NaiveBayes classifier function"""
	splitPoint = int(0.8*len(doc))
	xTrain = doc[:splitPoint]
	xTest = doc[splitPoint:]
	yTrain = age[:splitPoint]
	yTest = age[splitPoint:]

	vec = TfidfVectorizer(preprocessor = preprocessing, tokenizer = None, sublinear_tf = True)
	classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
	classifier.fit(xTrain, yTrain)
	yGuess = classifier.predict(xTest)

	print(classification_report(yTest, yGuess))

def crossvalidationClassifier(x, y):
	vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)
	classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
	
	crossvalidatorPrecision = cross_val_score(classifier, x, y=y, cv=10, n_jobs=-1, scoring="precision_weighted")
	crossvalidatorRecall = cross_val_score(classifier, x, y=y, cv=10, n_jobs=-1, scoring="recall_weighted")
	crossvalidatorFscore = cross_val_score(classifier, x, y=y, cv=10, n_jobs=-1, scoring="f1_weighted")

	print("Precision:", sum(crossvalidatorPrecision)/len(crossvalidatorPrecision))
	print("Recall:", sum(crossvalidatorRecall)/len(crossvalidatorRecall))
	print("F1-score:", sum(crossvalidatorFscore)/len(crossvalidatorFscore))

if __name__ == '__main__':
	languages = ['dutch', 'english', 'italian', 'spanish']
	dutDOC, dutGEN, dutAGE, dutAUT = fileRead(languages[0])
	engDOC, engGEN, engAGE, engAUT = fileRead(languages[1])
	itaDOC, itaGEN, itaAGE, itaAUT = fileRead(languages[2])
	spaDOC, spaGEN, spaAGE, spaAUT = fileRead(languages[3])


	#crossvalidationClassifier(spaDOC, spaAGE)
	naiveBayesClassfier(engDOC, engGEN, engAGE, engAUT)
	










"""
	allDOC = []
	allGEN = []
	allAGE = []
	allAUT = []
	allDOC.extend(dutDOC)
	allDOC.extend(engDOC)
	allDOC.extend(itaDOC)
	allDOC.extend(spaDOC)
	allGEN.extend(dutGEN)
	allGEN.extend(engGEN)
	allGEN.extend(itaGEN)
	allGEN.extend(spaGEN)
	allAGE.extend(dutAGE)
	allAGE.extend(engAGE)
	allAGE.extend(itaAGE)
	allAGE.extend(spaAGE)"""

	


