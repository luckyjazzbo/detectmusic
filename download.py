import os
import re
import requests
import certifi
import chardet
from w3lib.encoding import html_to_unicode


USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) ' \
             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36'


def _guess_encoding(body):
    return chardet.detect(body).get('encoding')


def get_with_cache(url):
    cache_key = re.sub(r'[^a-zA-Z0-9_-]', '', url)

    if os.path.isfile('cache/' + cache_key):
        body = open('cache/' + cache_key, 'r', encoding="utf-8").read()
    else:
        try:
            response = requests.get(url,
                                    headers={'user-agent': USER_AGENT},
                                    verify=certifi.where())
            detected_encoding, body = html_to_unicode(
                response.headers.get('content-type'),
                response.content,
                default_encoding='utf8',
                auto_detect_fun=_guess_encoding,
            )
        except requests.RequestException as e:
            print('Failed to get {url} - {error}'.format(url=url, error=str(e)))
            body = ''

        open('cache/' + cache_key, 'w', encoding="utf-8").write(body)

    return body
