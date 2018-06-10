# __*__ coding: UTF-8 __*__
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

def turnToWordList(sentence):
	#去掉HTML标签，获得内容
	text = BeautifulSoup(sentence, "html.parser").get_text()
	#print("text1:", text)
	text = re.sub("[^a-zA-Z]", " ", text)
	#print("text2:", text)
	words = text.lower().split()
	return words

#数据的读入

train = pd.read_csv('labeledTrainData.tsv', header = 0, delimiter="\t", quoting = 3)
test = pd.read_csv('testData.tsv', header = 0, delimiter="\t", quoting = 3)

trainData = []
for i in range(len(train['review'])):
	trainData.append(' '.join(turnToWordList(train['review'][i])))
print(len(trainData))

testData = []
for i in range(len(test['review'])):
	testData.append(' '.join(turnToWordList(test['review'][i])))
print(len(testData))


#print(train.head())
#print(test.head())

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
print('training TFIDF')
tfidf = TFIDF(min_df=2, max_features=1000, strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1,3), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words="english")

allData = trainData + testData
lentrain = len(trainData)

tfidf.fit(allData)
allData = tfidf.transform(allData)

train_x = allData[:lentrain]
test_x = allData[lentrain:]

print("TF-IDF处理结束")

# fout = open("vectorData.txt", "w")

# for i in range(lentrain):
#	print(train['sentiment'][i], train_x.data[i], trainData[i], file = fout)

import json
import datetime
def predict_save(method, path):
	print(path)
	
	begin = datetime.datetime.now()
	predictResult = np.array(method.predict(test_x)).tolist()
	end = datetime.datetime.now()
	k = end - begin
	print('时间：', k.total_seconds())

	print('保存结果...')
	fout = open(path, 'w')
	print(json.dumps(predictResult), file = fout)
	print('结束.')


def m(path):
	reader = csv.reader(open('nb_output.csv', 'r'))
	ids = []
	for m in reader:
		ids.append(m[0])
		f = open(path, 'r')
		line = json.loads(f.readline())
		rows = []
		for i in range(len(line)):
			rows.append([ids[i], line[i]])
			out = open(path.split('.')[0] + '.csv','a', newline='')
			csv_write = csv.writer(out,dialect='excel')
			csv_write.writerow(['id', 'sentiment'])
			for v in rows:
   				csv_write.writerow(v)
			out.close()


#利用朴素贝叶斯算法进行分类
from sklearn.naive_bayes import MultinomialNB as MNB


label = train['sentiment']


MNBmodle = MNB(alpha=1.0, class_prior=None, fit_prior=True)
svm_model = LinearSVC() #SVM
knn = KNeighborsClassifier() #K邻近
mlp=MLPClassifier(hidden_layer_sizes=(30,30,30),activation='logistic',max_iter=100) #感知机
clf = tree.DecisionTreeClassifier(criterion='gini')


print('train MNB')
begin = datetime.datetime.now()
MNBmodle.fit(train_x, label)
end = datetime.datetime.now()
k = end - begin
print('MNB训练时长：', k.total_seconds())

predict_save(modle, 'MNB.json')
m('MNB.json')


print('train SVM')

begin = datetime.datetime.now()
svm_model.fit(train_x, label)
end = datetime.datetime.now()
k = end - begin
print('MNB训练时长：', k.total_seconds())

predict_save(svm_model, 'svm.json')
m('svm.json')

print('train knn')

begin = datetime.datetime.now()
knn.fit(train_x, label)
end = datetime.datetime.now()
k = end - begin
print('训练时长：', k.total_seconds())

predict_save(knn, 'knn.json')
m('knn.json')


print('train 感知机')

begin = datetime.datetime.now()
mlp.fit(train_x, label)
end = datetime.datetime.now()
k = end - begin
print('训练时长：', k.total_seconds())

predict_save(mlp, 'mlp.json')
m('mlp.json')


print('train 决策树')

begin = datetime.datetime.now()
clf.fit(train_x, label)
end = datetime.datetime.now()
k = end - begin
print('MNB训练时长：', k.total_seconds())

predict_save(clf, 'clf.json')
m('clf.json')


