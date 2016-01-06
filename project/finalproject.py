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
	ageDict = {'XX-XX':'0', '18-24':'1', '25-34':'2', '35-49':'3', '50-XX':'4'}
	for xmlfile in fileList:
		tree = ET.parse(xmlfile)
		root = tree.getroot()
		for child in root:
			documents.append(child.text)
			genders.append(informationDict[root.attrib['id']][0])
			ages.append(ageDict[informationDict[root.attrib['id']][1]])
			authors.append(root.attrib['id'])
	return documents, genders, ages, authors

def identity(x):
	"""Dummy function"""
	return x

def naiveBayesClassfier(doc, gen, age, aut):
	splitPoint = int(0.80*len(doc))
	xTrain = doc[:splitPoint]
	yTrain = gen[:splitPoint]
	xTest = doc[splitPoint:]
	yTest = gen[splitPoint:]

	vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)
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

	#naiveBayesClassfier(engDOC, engGEN, engAGE, engAUT)
	
	crossvalidationClassifier(spaDOC, spaAGE)


