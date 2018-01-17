import re


class Example:
    def __init__(self, url, category, title='', h1='', purpose='train'):
        self.url = url
        self.category = category
        self.title = title
        self.h1 = h1
        self.purpose = purpose

    def keywords(self):
        text = '{} {} {}'.format(self.clean_url(), self.title, self.h1)
        return re.sub(r'\s+', ' ', text)

    def clean_url(self):
        res = re.sub(r'^https?://[^/]+', '', self.url)
        res = re.sub(r'\.html$', '', res)
        res = re.sub(r'[^\w\d]', ' ', res)
        res = re.sub(r'\d+', ' num ', res)
        res = re.sub(r'_', ' ', res)
        return res
