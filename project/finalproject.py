#!usr/bin/python3.4

import sys, os
import glob
from collections import defaultdict
import xml.etree.ElementTree as ET

def fileRead(language):
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
		print(root.attrib['id'])
		for child in root:
			documents.append(child.text)
			genders.append(informationDict[root.attrib['id']][0])
			ages.append(informationDict[root.attrib['id']][1])
			authors.append(root.attrib['id'])
		
	return documents, genders, ages, authors

if __name__ == '__main__':
	languages = ['dutch', 'english', 'italian', 'spanish']
	if len(sys.argv) == 2 and sys.argv[1] in languages:
		language = sys.argv[1]
		fileRead(language)

	else:
		print("Usage: python finalproject.py <language>", file=sys.stderr)
		print("Accepted languages are 'dutch', 'english', 'italian' and 'spanish'", file=sys.stderr)