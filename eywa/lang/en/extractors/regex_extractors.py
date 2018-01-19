# -*- coding: utf-8 -*-
from ....entities import Email, PhoneNumber, Url
from .extractor import RegexExtractor
import re


email_regex = "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
phone_regex = "(\+?[0-9\(][0-9\- \(\)\.]{6,16}( ?e?xt?\.? ?\d+)?)"
url_regex = "((https?://|ftp://|www\.|[^\s:=]+@www\.).*?[a-z_\/0-9\-\#=&])(?=(\.|,|;|\?|\!)?(\"|'|«|»|\[|\s|\r|\n|$))"


class EmailExtractor(RegexExtractor):
    def __init__(self):
        super(EmailExtractor, self).__init__(email_regex, Email)


class PhoneNumberExtractor(RegexExtractor):
     def __init__(self):
        super(PhoneNumberExtractor, self).__init__(phone_regex, PhoneNumber)


class UrlExtractor(RegexExtractor):
     def __init__(self):
        super(UrlExtractor, self).__init__(url_regex, Url)
