from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
twenty_train.target_names
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']

#print(len(twenty_train.data))
#print("\n".join(twenty_train.data[3].split("\n")[:5]))

#CountVectorizer  methodu
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
#print(X_train_counts.shape)
#print(count_vect.vocabulary_.get(u'set'))
###


#TfidfTransformer methodu
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)
####

#naif bayes sınıflandırıcı
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
###

#en genel çözüm
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
print(text_clf.fit(twenty_train.data, twenty_train.target))
####


#naif bayese göre işlem ne kadar verimli
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print('Naif Bayes e göre doğruluk: ', np.mean(predicted == twenty_test.target))
##########

#doğruluğu SVM e göre yaparsak
text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))])
print(text_clf.fit(twenty_train.data, twenty_train.target))

predicted = text_clf.predict(docs_test)
print('SVM e göre doğruluk: ', np.mean(predicted == twenty_test.target))
##########

#tablo halinde metric görünümü
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
