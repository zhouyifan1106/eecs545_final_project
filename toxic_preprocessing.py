import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import csv
import json
import numpy as np


y = []
x = [] 
training_words = []
stop_words = set(stopwords.words('english'))
count= 0

# Select top 10000 samples, perform nltk operations
with open('train_github.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
    	if count < 10000:
	    	text = row[1]
	    	text = re.sub("[0-9]+", " ", text)
	    	text = nltk.word_tokenize(text)
	    	text = [word.lower() for word in text]
	    	text = [word for word in text if word.isalpha()]
	    	text = [word for word in text if not word in stop_words]
	    	x.append(text)
	    	training_words = training_words + [word for word in text if word not in training_words]
	    	y.append(row[2:8])
	    	count += 1
	    	print (count)

x_train = x

# Word frequency
fd = nltk.FreqDist(training_words)
# Print Top Feature Outcome
for x in [100,200,500,1000,1500,2000,2500,3000,3500,4000]:
	top = fd.most_common(x)
	top_words = []
	for i in range(0,len(top)):
		top_words.append(top[i][0])
	i = 0
	# check if all training sample has at least one word listed as feature
	for sample in x_train:
		if not (any(x in top_words for x in sample)):
			#print ("Need more features")
			i += 1
	print (str(i) + " training samples do not have any word in the feature matrix with feature number " + str(x))


# Decide to use 2000 features for training
number_of_features = 1000
top = fd.most_common(number_of_features)
top_words = []
for i in range(0,len(top)):
	top_words.append(top[i][0])


# construct counting matrix
x_train_counting = np.zeros((len(x_train),number_of_features))
for row in range(0,len(x_train)):
	for col in range(0,number_of_features):
		x_train_counting[row][col] = x_train[row].count(top_words[col])

x = x_train_counting
y = np.array(y).astype("int")

np.savetxt("x_10000sample_2000.csv", x_train_counting, delimiter=",")
np.savetxt("y_10000sample_2000.csv", y, delimiter=",",fmt='%i')









