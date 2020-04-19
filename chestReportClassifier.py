import os
import sys
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn import metrics
import _pickle as cPickle
currentDir=os.path.abspath(os.path.dirname(__file__))

#Read top 5 rows from the training data set
#Use os.currentdir instead of hard coded path
df = pd.read_csv("F:\\IMPORTANT_DATA\\chest_diagnosis.csv", encoding = "ISO-8859-1")


##Get the rows other than id and findings
#df_diagnosis = df.drop(['id', 'Findings'], axis=1)
##print (df_diagnosis)
#
#counts = []
##Get column names 
#categories = list(df_diagnosis.columns.values)
##print (categories)
#
###Create tuples containing category names and number of abnormal cases per category (Add Column wise )
#for i in categories:
#    counts.append((i, df_diagnosis[i].sum()))
##print (counts)
#
###Print the table
#df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
##print (df_stats)
#
###Bar chart
#df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=True, figsize=(8, 5))
#plt.title("Number of findings per category",fontsize=10)
#plt.ylabel('Number of Occurrences', fontsize=10)
#plt.xlabel('category', fontsize=10)
#plt.show(block=True)
#
##
##Read each rows from 2nd Column & Add Row wise
#rowsums = df.iloc[:,2:].sum(axis=1)
##print (rowsums)
#
##Get number of 0 , 1 & 2 in the above list
#x=rowsums.value_counts()
##print (x)
#
##plot
#plt.figure(figsize=(8,5))
#ax = sns.barplot(x.index, x.values)
#plt.title("Multiple categories per Findings")
#plt.ylabel('Number of Occurrences', fontsize=10)
#plt.xlabel('Number of categories', fontsize=10)
#plt.show(block=True)
#
#
#print('Percentage of reports that are Normal & Contains only chest region')
#print(len(df[(df['Abnormal_Lungs']==0) & (df['Abnormal_Heart']==0) & (df['Damaged_Ribs']==0) & (df['Other_Region']== 0)]) / len(df))

#print('Number of missing comments in comment text:')
#print (df['Findings'].isnull().sum())

try:
#Data PreProcessing
    def clean_text(text):
       
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')
        return text
except Exception as e:
    print (e)
    
#Stop Word Remover
stop_words = [x.strip() for x in open('F:\\stopWords.txt','r').read().split('\n')]

#Stemming
porter_stemmer = PorterStemmer()
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words


df['Findings'] = df['Findings'].map(lambda com : clean_text(com))



categories = ['Abnormal_Lungs', 'Abnormal_Heart', 'Damaged_Ribs', 'Other_Region']
# test data can be called as dev_data during validation
train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)

X_train = train.Findings
X_test = test.Findings


docs_new = ['AP view radiograph of chest shows ET tube in situ. Hazziness seen in lower zone of right lung. Cardio-mediastinal shadows have normal contour.']
NB_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words=stop_words,analyzer='word',ngram_range=(1,3),tokenizer = stemming_tokenizer,norm='l2',use_idf=True, smooth_idf=True)),
                        ('clf', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))),])



#SGDClassifier(loss='perceptron', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
#MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
#LogisticRegression(random_state=42, solver='lbfgs',multi_class='multinomial')




for category in categories:
    # training the model
    NB_pipeline.fit(X_train, train[category])
    
    #Test Accuracy
    prediction = NB_pipeline.predict(X_test)
    print (category)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print ('\n')        
            
    if (os.path.isdir(os.path.join(currentDir,"Models"))):
        with open(os.path.join(currentDir,"Models",category+'.pkl'), 'wb') as fid:
            cPickle.dump(NB_pipeline, fid)
            
    else:
        os.mkdir(currentDir+"\\"+"Models")
        with open(os.path.join(currentDir,"Models",category+'.pkl'), 'wb') as fid:
            cPickle.dump(NB_pipeline, fid)
                  
    
    #with open(os.path.join(currentDir,"Models",category+'.pkl'),'rb') as pid:
    #    gnb_loaded=cPickle.load(pid)
     
    #prediction = gnb_loaded.predict(X_test)
    #print('Test accuracy with saved model {}'.format(accuracy_score(test[category], prediction)))   
    
    
    #Validation Accuracy
    # if there is no seperate test data, train data divided on cv and checked.
    scores=cross_val_score(NB_pipeline,X_train,train[category],cv=5)
    print ("Validation accuracy at each iteration %s"%(str(scores)))
    print("Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print ('\n')
    
    #Metrices on test dataset:
    test_target_names=[]
    testSet=set(test[category])
   
    test_target_list= list(testSet)
    refCategory=['Normal','Abnormal']
    for eachValue in test_target_list:
       test_target_names.append(refCategory[eachValue])
    
    
    print(metrics.classification_report(test[category], prediction,target_names=test_target_names))
    print ("Confusion Matrix %s"%(str(metrics.confusion_matrix(test[category], prediction))))
    print ('--------------------------------------------------\n')
    
    

    



