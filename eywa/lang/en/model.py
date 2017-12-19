# -*- coding: utf-8 -*-

from .filenames import vectors_file_name, vector_index_file_name, interrupt_flag_file_name
from .filenames import vocab_db_file_name, inverse_vocab_db_file_name
from .filenames import frequency_file_name, frequency_db_file_name
from.filenames import phrases_db_file_name, tokens_db_file_name
from .database import Database
from . import indexer
from numpy.core.umath_tests import inner1d
import numpy as np
import annoy



indexer.run()

annoy_index = annoy.AnnoyIndex(300, 'angular')
annoy_index.load(vector_index_file_name)
vocab_db = Database(vocab_db_file_name)
inverse_vocab_db = Database(inverse_vocab_db_file_name)
frequency_db = Database(frequency_db_file_name)
phrases_db = Database(phrases_db_file_name)
frequency_db = Database(frequency_db_file_name)


def tokenizer(X):
    return [x.strip() for x in re.split('(\W+)?', X) if x.strip()]


def split_into_sentences(text):
    orig_text = text
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    if not sentences:
        return [orig_text]
    return sentences


_PHRASER_THRESHOLD = 0.0004
_IGNORE_PHRASES = ['tell_me']


def phraser(X):
    num_x = len(X)
    i = 0
    output = []
    while i < num_x:
        x = X[i]
        phrases = phrases_db[x]
        phrases = [vocab_db[p] for p in phrases]
        best_phrase = None
        best_phrase_prob = 0
        best_phrase_len = 0
        for phrase in phrases:
            if phrase in _IGNORE_PHRASES:
                continue
            tokens = phrase.split('_')
            num_tokens = len(tokens)
            if num_tokens + i > num_x:
                continue
            if phrase in frequency_db:
                freq_phrase = frequency_db[phrase]
            else:
                freq_phrase = 0
            freq_tokens = []
            for t in tokens:
                if t in frequency_db:
                    freq_tokens.append(frequency_db[t])
                else:
                    freq_tokens.append(0)
            freq_tokens = np.mean(freq_tokens)
            prob = freq_phrase * num_tokens ** 3 / freq_tokens
            if prob > best_phrase_prob:
                best_phrase_prob = prob
                best_phrase = phrase
                best_phrase_len = num_tokens
        if best_phrase_prob >= _PHRASER_THRESHOLD:
            output.append(best_phrase)
            i += best_phrase_len
        else:
            output.append(x)
            i += 1
    return output


def get_embedding(word, sense_disambiguation='max', normalize=True, default=0):
    if '|' not in word:
        if word in tokens_db:
            tokens_idxs = tokens_db[word]
            if not tokens_idxs:
                if default == 0:
                    emb = np.zeros(300)
                elif default is None:
                    emb = None
            elif sense_disambiguation == 'max':
                emb = annoy_index.get_item_vector(tokens_idxs[0])
            elif sense_disambiguation == 'avg':
                emb = np.mean([annoy_index.get_item_vector(i) for i in tokens_idxs], 0)
            else:
                emb = None
                for tidx in tokens_idxs:
                    sense = vocab_db[tidx].split('|')[1]
                    if sense == sense_disambiguation:
                        emb = annoy_index.get_item_vector(tidx)
                        break
                if emb is None:
                    emb = annoy_index.get_item_vector(tokens_idxs[0])
        elif '_' in word:
            embs = []
            sub_tokens = word.split('_')
            for t in sub_tokens:
                t_emb = get_embedding(t, sense_disambiguation, False, None)
                if t_emb is not None:
                    embs.append(t_emb)
            emb = np.mean(embs, 0)
        elif word in phrases_db:
            phrases_idxs = phrases_db[word]
            if sense_disambiguation == 'max':
                emb = annoy_index.get_item_vector(phrases_idxs[0])
            elif sense_disambiguation == 'avg':
                emb = np.mean([annoy_index.get_item_vector(i) for i in phrases_idxs], 0)
            else:
                emb = None
                for pidx in phrases_idxs:
                    p = vocab_db[pidx]
                    if p.split('|')[1] == sense_disambiguation:
                        emb = annoy_index.get_item_vector(pidx)
                if emb is None:
                    emb = np.mean([annoy_index.get_item_vector(i) for i in phrases_idxs], 0)
    elif word in inverse_vocab_db:
        word_index = inverse_vocab_db[word]
        emb = annoy_index.get_item_vector(word_index)
    else:
        word, sense = word.split('|')
        emb = get_embedding(word, sense, False)
    if emb is not None and normalize:
        mag = inner1d(emb, emb)
        if mag != 0:
            emb /= mag
    return emb


def get_frequency(word, sense_disambiguation='max'):
    if '|' not in word:
        if word in tokens_db:
            tokens_idxs = tokens_db[word]
            if not tokens_idxs:
                if default == 0:
                    freq = 0
            elif sense_disambiguation == 'max':
                freq = frequency_db[tokens_idxs[0]]
            elif sense_disambiguation == 'avg':
                freq = np.mean([frequency_db[i] for i in tokens_idxs])
            else:
                freq = None
                for tidx in tokens_idxs:
                    sense = vocab_db[tidx].split('|')[1]
                    if sense == sense_disambiguation:
                        freq = frequency_db[tidx]
                        break
                if freq is None:
                    freq = frequency_db[tokens_idxs[0]]
        elif '_' in word:
            first_token = word.split('_')[0]
            freq = get_frequency(first_token, sense_disambiguation)
        elif word in phrases_db:
            phrases_idxs = phrases_db[word]
            if sense_disambiguation == 'max':
                freq = frequency_db[phrases_idxs[0]]
            elif sense_disambiguation == 'avg':
                freq = np.mean([frequency_db[i] for i in tokens_idxs])
            else:
                freq = None
                for pidx in phrases_idxs:
                    p = vocab_db[pidx]
                    if p.split('|')[1] == sense_disambiguation:
                        freq = frequency_db[i]
                if freq is None:
                    freq = np.mean([frequency_db[i] for i in phrases_idxs])
    elif word in inverse_vocab_db:
        word_index = inverse_vocab_db[word]
        freq = frequency_db[word_index]
    else:
        word, sense = word.split('|')
        freq = get_frequency(word, sense, False)
    return freq


class Token(object):
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    @property
    def embedding(self):
        if not hasattr(self, '_embedding'):
            self._embedding = get_embedding(self.text.lower(), default=None)
        if self._embedding is None:
            return np.zeros(300)
        return self._embedding

    @property
    def frequency(self):
        if not hasattr(self, '_frequency'):
            self._frequency = get_frequency(self.text.lower())
        return self._frequency


class Document(object):
    def __init__(self, text):
        self.text = text
        self.tokens = [Token(w) for w in phraser(tokenizer(text))]

    def __iter__(self):
        self.iter_index = 0
        return self

    def next(self):
        if self.iter_index == len(self.tokens):
            raise StopIteration()
        token = self.tokens[self.iter_index]
        self.iter_index += 1
        return token

    def __getitem__(self, key):
        if type(key) is int:
            return self.tokens[key]
        else:
            for token in tokens:
                if token.text == key:
                    return token
            raise IndexError('Token not found ' + key)
 
    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return self.text

    @property
    def embedding(self):
        if not hasattr(self, '_embedding'):
            self._embedding = np.mean([t.embedding for t in self.tokens], 0)
        return self._embedding
