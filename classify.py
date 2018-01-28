from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from random import shuffle
from bs4 import BeautifulSoup
from website import Website
import csv
import download


def prepare_training_data():
    websites = []

    with open('music_urls_2.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == '3':
                website = Website(row[0], 0)
            elif row[2] == '2':
                website = Website(row[0], 1)
            elif row[1] == '1':
                website = Website(row[0], 2)
            else:
                raise ValueError('Invalid row - {}'.format(str(row)))

            body = download.get_with_cache(website.url)

            if len(body) > 0:
                soup = BeautifulSoup(body, 'html.parser')
                website.title = soup.title.string
                headers = soup.find_all('h1')
                if len(headers) > 0:
                    website.h1 = str(headers[0].text)
                websites.append(website)

    shuffle(websites)

    test_size = int(0.4 * len(websites))
    for website in websites[-test_size:]:
        website.purpose = 'test'

    return websites


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

    train_words = [ex.keywords() for ex in data if ex.purpose == 'train']
    train_categories = [ex.category for ex in data if ex.purpose == 'train']
    test_words = [ex.keywords() for ex in data if ex.purpose == 'test']
    test_categories = [ex.category for ex in data if ex.purpose == 'test']

    classifier = Classifier()
    classifier.learn_features(train_words)
    classifier.learn(train_words, train_categories)

    prediction = classifier.predict(test_words)
    f1 = f1_score(test_categories, list(prediction), average='micro')

    print('F1 score: {}'.format(f1))

    for predicted, real, example in zip(list(prediction), test_categories, [ex for ex in data if ex.purpose == 'test']):
        if predicted != real:
            print(predicted, real, example.url, example.keywords())
