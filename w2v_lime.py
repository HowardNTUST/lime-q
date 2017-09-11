#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:13:33 2017

@author: slave1
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import jieba
from load_datasets import LoadDataset
from data_trusting import get_classifier
import embedding_forest

 
def get_classifier(name, vectorizer):
  if name == 'logreg':
    return linear_model.LogisticRegression(fit_intercept=True)
  if name == 'random_forest':
    return ensemble.RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=5, n_jobs=10)
  if name == 'svm':
    return svm.SVC(probability=True, kernel='rbf', C=10,gamma=0.001)
  if name == 'tree':
    return tree.DecisionTreeClassifier(random_state=1)
  if name == 'neighbors':
    return neighbors.KNeighborsClassifier()
  if name == 'embforest':
    return embedding_forest.EmbeddingForest(vectorizer)



train_data, train_labels, test_data, test_labels, class_names = LoadDataset()
vectorizer = TfidfVectorizer(lowercase=False, binary=True, ngram_range=(1,3)) 
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)
#terms = np.array(list(vectorizer.vocabulary_.keys()))
#indices = np.array(list(vectorizer.vocabulary_.values()))
#inverse_vocabulary = terms[np.argsort(indices)]

np.random.seed(1)
classifier = get_classifier('embforest', vectorizer)
classifier.fit(train_vectors, train_labels)


pred = classifier.predict(test_vectors)
sklearn.metrics.f1_score(test_labels, pred, average='binary')


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, classifier)
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)
c.predict_proba([newsgroups_test.data[idx]])[0,1]


idx = 83
exp = explainer.explain_instance(test_data[idx],  c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([test_data[idx]])[0,1])
print('True class: %s' % class_names[test_labels[idx]])
exp.as_list()
exp.show_in_notebook(text=False)
exp.save_to_file('oi.html')

print('Original prediction:', classifier.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', classifier.predict_proba(tmp)[0,1])
print('Difference:', classifier.predict_proba(tmp)[0,1] - classifier.predict_proba(test_vectors[idx])[0,1])
