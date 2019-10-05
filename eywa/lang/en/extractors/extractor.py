from ....entities import Entity
import re


class Extractor(object):
    def __init__(self):
        pass

    def __call__(self, x):
        # return [()]
        pass


class RegexExtractor(Extractor):

    def __init__(self, regex, entity_type=Entity):
        super().__init__()
        self.regex = regex
        self.entity_type = entity_type
        self.compiled_regex = re.compile(regex, re.IGNORECASE)

    def __call__(self, text):
        y = []
        matches = self.compiled_regex.finditer(text)
        for match in matches:
            start, end = match.start(), match.end()
            string = text[start: end]
            y.append(((start, end), self.entity_type(string, string)))
        return y
