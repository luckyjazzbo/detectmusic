from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from random import shuffle
from bs4 import BeautifulSoup
from page import Page
import csv
import download


TEST_SIZE = 0.3
META_LEARNING_SIZE = 0.4


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
                page.title = str(soup.title.string)
                headers = soup.find_all('h1')
                if len(headers) > 0:
                    page.h1 = str(headers[0].text)
                pages.append(page)

    shuffle(pages)

    test_size = int(TEST_SIZE * len(pages))
    for p in pages[-test_size:]:
        p.purpose = 'test'

    return pages


class Classifier:
    def __init__(self):
        stemmer = SnowballStemmer('russian')
        analyzer = CountVectorizer().build_analyzer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))

        self.url_vectorizer = CountVectorizer(tokenizer=stemmed_words, max_features=10_000)
        self.title_vectorizer = CountVectorizer(tokenizer=stemmed_words, max_features=10_000)
        self.h1_vectorizer = CountVectorizer(tokenizer=stemmed_words, max_features=10_000)

        self.url_classifier = MLPClassifier(alpha=0.1, max_iter=500)
        self.title_classifier = MLPClassifier(alpha=0.1, max_iter=500)
        self.h1_classifier = MLPClassifier(alpha=0.1, max_iter=500)

        self.meta_classifier = MLPClassifier(alpha=0.01)

    def fit(self, pages):
        number_for_meta_learning = int(META_LEARNING_SIZE * len(pages))
        base_pages = pages[:-number_for_meta_learning]

        self._fit_base_classifiers(base_pages)

        meta_pages = pages[-number_for_meta_learning:]
        meta_categories = [p.category for p in meta_pages]

        combined_prediction = self._make_and_combine_base_predictions(meta_pages)
        self.meta_classifier.fit(combined_prediction, meta_categories)

    def _fit_base_classifiers(self, pages):
        categories = [p.category for p in pages]

        url_features = self.url_vectorizer.fit_transform([p.clean_url() for p in pages])
        self.url_classifier.fit(url_features, categories)

        title_features = self.title_vectorizer.fit_transform([p.title for p in pages])
        self.title_classifier.fit(title_features, categories)

        h1_features = self.h1_vectorizer.fit_transform([p.h1 for p in pages])
        self.h1_classifier.fit(h1_features, categories)

    def predict(self, pages):
        combined_prediction = self._make_and_combine_base_predictions(pages)
        return self.meta_classifier.predict(combined_prediction)

    def _make_and_combine_base_predictions(self, pages):
        url_features = self.url_vectorizer.transform([p.clean_url() for p in pages])
        url_prediction = self.url_classifier.predict_proba(url_features)

        title_features = self.title_vectorizer.transform([p.title for p in pages])
        title_prediction = self.title_classifier.predict_proba(title_features)

        h1_features = self.h1_vectorizer.transform([p.h1 for p in pages])
        h1_prediction = self.h1_classifier.predict_proba(h1_features)

        combined_prediction = []
        for i in range(len(url_prediction)):
            row = []
            for j in url_prediction[i]: row.append(j)
            for j in title_prediction[i]: row.append(j)
            for j in h1_prediction[i]: row.append(j)
            combined_prediction.append(row)

        return combined_prediction


if __name__ == '__main__':
    data = prepare_training_data()

    train_pages = [p for p in data if p.purpose == 'train']
    test_pages = [p for p in data if p.purpose == 'test']

    classifier = Classifier()
    classifier.fit(train_pages)

    prediction = classifier.predict(test_pages)
    test_categories = [p.category for p in test_pages]
    f1 = f1_score(test_categories, list(prediction), average='micro')

    print('F1 score: {}'.format(f1))
    print('Misclassified pages:')
    for predicted, page in zip(list(prediction), test_pages):
        if predicted != page.category:
            print(predicted, '-', page.category, '-', page.url, '-', page.clean_title(), '-', page.clean_h1())
