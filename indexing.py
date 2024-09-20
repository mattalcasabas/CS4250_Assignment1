#-------------------------------------------------------------------------
# AUTHOR: Matthew Alcasabas
# FILENAME: indexing.py
# SPECIFICATION: performs stopword removal, stemming, and calculates tf-idf for each term
# FOR: CS 4250- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#Importing some Python libraries
import csv
# import pandas for table generation
import pandas as pd
# import math for log function
import math

documents = []
terms = []

#Reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

# print("Documents: ", documents)

# break each document into terms based off space
for document in documents:
    terms.append(document.split(' '))

# print("Terms: ", terms)
    

#Conducting stopword removal for pronouns/conjunctions. Hint: use a set to define your stopwords.
stopWords = ['i', 'and', 'she', 'her', 'they', 'their']
processedTerms = []

for termList in terms:
    filteredList = [term for term in termList if term.lower() not in stopWords]
    processedTerms.append(filteredList)
          
# print(processedTerms)


#Conducting stemming. Hint: use a dictionary to map word variations to their stem.
stemming = {'love': ['love', 'loves'], 'cat': ['cat', 'cats'], 'dog': ['dog', 'dogs']}
stemmedTerms = []

for termList in processedTerms:
    replacedList = []
    for term in termList:
        sub = None
        for k, v in stemming.items():
            if term.lower() in v:
                sub = k
                break
        replacedList.append(sub if sub else term)
    stemmedTerms.append(replacedList)

# print(stemmedTerms)
    

#Identifying the index terms.
termCountList = []

for termList in stemmedTerms:
    termCount = {}

    for term in termList:
        if term in termCount:
            termCount[term] += 1
        else:
            termCount[term] = 1

    termCountList.append(termCount)
# print(termCountList)
# generate and display pandas dataframe
indexTerms = pd.DataFrame(termCountList).fillna(0).astype(int)

# print(indexTerms)

#Building the document-term matrix by using the tf-idf weights.
#--> add your Python code here
tfList = []
idfDict = {}

for termList in stemmedTerms:
  termCount = {}
  totalTerms = len(termList)

  # calculate tf for each term
  for term in termList:
      termCount[term] = termCount.get(term, 0) + 1
  for term in termCount:
      termCount[term] /= totalTerms
  tfList.append(termCount)

totalDocs = len(stemmedTerms)

# find number of docs that contain each term
for termList in stemmedTerms:
    uniqueTerms = set(termList)
    for term in uniqueTerms:
        idfDict[term] = idfDict.get(term, 0) + 1

# calculate idf for each term
for term in idfDict:
    idfDict[term] = math.log(totalDocs / idfDict[term])

# calculate tf-idf
tfidfList = []
for i, termList in enumerate(tfList):
    tfidfDict = {}

    for term, tfVal in termList.items():
        tfidfDict[term] = tfVal * idfDict[term]
    tfidfList.append(tfidfDict)

#Printing the document-term matrix.
tfidfDataFrame = pd.DataFrame(tfidfList).fillna(0)
print(tfidfDataFrame)