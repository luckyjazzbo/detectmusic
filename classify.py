from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from random import shuffle
from bs4 import BeautifulSoup
from example import Example
import csv
import download


data = []

with open('music_urls_2.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[3] == '3':
            example = Example(row[0], 0)
        elif row[2] == '2':
            example = Example(row[0], 1)
        elif row[1] == '1':
            example = Example(row[0], 2)
        else:
            raise ValueError('Invalid row - {}'.format(str(row)))

        body = download.get_with_cache(example.url)

        if len(body) > 0:
            soup = BeautifulSoup(body, 'html.parser')
            example.title = soup.title.string
            headers = soup.find_all('h1')
            if len(headers) > 0:
                example.h1 = str(headers[0].text)
            data.append(example)


shuffle(data)
train_size = int(0.6 * len(data))

for example in data[train_size:]:
    example.purpose = 'test'


train_features = [ex.keywords() for ex in data if ex.purpose == 'train']
train_categories = [ex.category for ex in data if ex.purpose == 'train']
test_features = [ex.keywords() for ex in data if ex.purpose == 'test']
test_categories = [ex.category for ex in data if ex.purpose == 'test']

print([(ex.url, ex.keywords(), ex.category) for ex in data])


stemmer = SnowballStemmer('russian')
analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


vectorizer = CountVectorizer(tokenizer=stemmed_words, max_features=10_000)
train_X = vectorizer.fit_transform(train_features)

classifier = MLPClassifier(alpha=1, max_iter=500)
classifier.fit(train_X, train_categories)
test_X = vectorizer.transform(test_features)
prediction = classifier.predict(test_X)

f1 = f1_score(test_categories, list(prediction), average='micro')
print('F1 score: {}'.format(f1))

for predicted, real, example in zip(list(prediction), test_categories, [ex for ex in data if ex.purpose == 'test']):
    if predicted != real:
        print(predicted, real, example.url, example.keywords())
