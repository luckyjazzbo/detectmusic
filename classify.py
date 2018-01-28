from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from random import shuffle
from bs4 import BeautifulSoup
from page import Page
import csv
import download


def prepare_training_data():
    pages = []

    with open('music_urls_2.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == '3':
                page = Page(row[0], 0)
            elif row[2] == '2':
                page = Page(row[0], 1)
            elif row[1] == '1':
                page = Page(row[0], 2)
            else:
                raise ValueError('Invalid row - {}'.format(str(row)))

            body = download.get_with_cache(page.url)

            if len(body) > 0:
                soup = BeautifulSoup(body, 'html.parser')
                page.title = soup.title.string
                headers = soup.find_all('h1')
                if len(headers) > 0:
                    page.h1 = str(headers[0].text)
                pages.append(page)

    shuffle(pages)

    test_size = int(0.4 * len(pages))
    for p in pages[-test_size:]:
        p.purpose = 'test'

    return pages


class Classifier:
    def __init__(self):
        stemmer = SnowballStemmer('russian')
        analyzer = CountVectorizer().build_analyzer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))

        self.vectorizer = CountVectorizer(tokenizer=stemmed_words, max_features=10_000)
        self.classifier = MLPClassifier(alpha=1, max_iter=500)

    def learn_features(self, words):
        return self.vectorizer.fit(words)

    def get_features(self, words):
        return self.vectorizer.transform(words)

    def learn(self, words, categories):
        self.classifier.fit(self.get_features(words), categories)

    def predict(self, words):
        return self.classifier.predict(self.get_features(words))


if __name__ == '__main__':
    data = prepare_training_data()

    train_words = [p.keywords() for p in data if p.purpose == 'train']
    train_categories = [p.category for p in data if p.purpose == 'train']
    test_words = [p.keywords() for p in data if p.purpose == 'test']
    test_categories = [p.category for p in data if p.purpose == 'test']

    classifier = Classifier()
    classifier.learn_features(train_words)
    classifier.learn(train_words, train_categories)

    prediction = classifier.predict(test_words)
    f1 = f1_score(test_categories, list(prediction), average='micro')

    print('F1 score: {}'.format(f1))

    for predicted, real, page in zip(list(prediction), test_categories, [p for p in data if p.purpose == 'test']):
        if predicted != real:
            print(predicted, real, page.url, page.keywords())
