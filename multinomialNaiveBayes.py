import sklearn
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


#Loading data set
categories = ['ABNORMAL_CHEST', 'NORMAL_CHEST']
Train_Set=sklearn.datasets.load_files("F://IMPORTANT_DATA//Chest_Reports", description=None, categories=categories, load_content=True, shuffle=True, encoding=None, decode_error='strict', random_state=42)
X_train, X_test, y_train, y_test = train_test_split(Train_Set.data, Train_Set.target, test_size=.1,random_state=42)
print ("Target Classes::"+str(Train_Set.target_names))

#Loading stop words from file
stop_words = [x.strip() for x in open('stopWords.txt.txt','r').read().split('\n')]
porter_stemmer = PorterStemmer()
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words


#STEP 1
#count_vect = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer='word',stop_words=stop_words,ngram_range=(1,2),tokenizer= stemming_tokenizer,min_df=0.7)
#X_train_counts = count_vect.fit_transform(X_train)


#STEP 2
#Directly using tfidf vectorizer
#tf = TfidfVectorizer(use_idf=True,norm='l1',smooth_idf=True,analyzer='word',stop_words=stop_words,ngram_range=(1,2),strip_accents='unicode',decode_error='ignore',min_df=0.6)
#txt_fitted = tf.fit(X_train)
#txt_transformed = txt_fitted.transform(X_train)
#print (tf.vocabulary_)
#print (txt_transformed.toarray())
#print (tf.idf_)


#STEP 3
#clf = MultinomialNB().fit(X_train_tf, y_train)
#docs_new = ['OpenGL on the GPU is fast']
#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = tf_transformer.transform(X_new_counts)

#predicted = clf.predict(X_new_tfidf)
#for doc, category in zip(docs_new, predicted):
#    print('%r => %s' % (doc, Train_Set.target_names[category]))




#View Idf of all the feattures    
#idf=tf.idf_
#featureIdf = dict(zip(txt_fitted.get_feature_names(), idf))
#token_weight = pd.DataFrame.from_dict(featureIdf, orient='index').reset_index()
#token_weight.columns=('token','weight')
#token_weight = token_weight.sort_values(by='weight', ascending=False)
#sns.barplot(x='token', y='weight', data=token_weight)            
#plt.title("Inverse Document Frequency(idf) per token")
#fig=plt.gcf()
#fig.set_size_inches(10,5)
#plt.show()
##




#Training and Prediction
docs_new = ['Lung fields appear normal.\
Both hila and vascular markings are normal.\
Mediastinum is central.\
No cardiomegaly.\
Right hemidiaphragmis elevated with mild blunting of the right CP angle.\
Rib cage and spine appears normal.'] #user input/ Report Content from RISPACS
text_clf = Pipeline([('vect', CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer='word',stop_words=stop_words,ngram_range=(1,3),tokenizer = stemming_tokenizer)),
                    ('tfidf', TfidfTransformer(use_idf=True,norm=None,smooth_idf=True)),('clf', MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)),])
text_clf.fit(X_train, y_train)

#Predicion of the input text
predicted=text_clf.predict(docs_new)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, Train_Set.target_names[category]))
    
    
#Validation
scores=cross_val_score(text_clf,X_train,y_train,cv=5)
print ("Validation accuracy at each iteration %s"%(str(scores)))
print("Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Test accuracy
docs_test = X_test
predicted = text_clf.predict(docs_test)
print ("Test Accuracy %s"%(str(np.mean(predicted == y_test))))


#Metrices on test dataset:
test_target_names=[]
testSet=set(y_test)
test_target_list= list(testSet)
for eachValue in test_target_list:
    test_target_names.append(Train_Set.target_names[eachValue])


print(metrics.classification_report(y_test, predicted,target_names=test_target_names))
print ("Confusion Matrix %s"%(str(metrics.confusion_matrix(y_test, predicted))))


#Tuning Parameter
parameters = {'vect__ngram_range': [(1, 1), (1, 3)],'tfidf__use_idf': (True, False),'tfidf__norm':('l1','l2',None),'clf__alpha': (1.0,1.5)}    
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=1,refit=True)   
gs_clf = gs_clf.fit(X_train[:400],y_train[:400])

print (gs_clf.best_score_ )
for param_name in sorted(parameters.keys()):
   print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
