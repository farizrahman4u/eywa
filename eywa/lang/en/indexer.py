from ...utils import lang_en_embeddings_dir
from ...utils import ProgressBar
from .filenames import vectors_file_name, vector_index_file_name
from .filenames import vocab_file_name, vocab_db_file_name, inverse_vocab_db_file_name
from .filenames import frequency_file_name, frequency_db_file_name
from .filenames import phrases_db_file_name, tokens_db_file_name
from .filenames import interrupt_flag_file_name as flag_file
from .filenames import vector_size_file_name
import numpy as np
import annoy
import os
import json
from .database import Database
from .download_embeddings import download


_required_files = [vector_index_file_name, frequency_db_file_name,
                   phrases_db_file_name, tokens_db_file_name,
                   vector_size_file_name, vocab_db_file_name,
                   inverse_vocab_db_file_name]


def _is_downloaded():
    return os.path.isfile(vectors_file_name)


def _is_corrupt():
    s = sum([os.path.isfile(f) for f in _required_files])
    return s > 0 and s < len(_required_files)
   

def _is_interrupted():
    return os.path.isfile(flag_file)


def _is_built():
    return all([os.path.isfile(f) for f in _required_files])

def _clear():
    for f in _required_files:
        if os.path.isfile(f):
            os.remove(f)

def _build():
    with open(flag_file, 'w') as f:
        f.write(' ')
    vectors = np.load(vectors_file_name)[1:]
    metric = 'angular'
    num_trees = 10
    dim = vectors.shape[1]
    with open(vector_size_file_name, 'w') as f:
        f.write(str(dim))
    print('Building tree...')
    annoy_index = annoy.AnnoyIndex(dim, metric)
    pbar = ProgressBar(len(vectors))
    for x in enumerate(vectors):
        annoy_index.add_item(*x)
        pbar.update()
    annoy_index.build(num_trees)
    annoy_index.save(vector_index_file_name)
    print('Done.')
    print('Creating databases...')
    vocab_db = Database(vocab_db_file_name)
    inverse_vocab_db = Database(inverse_vocab_db_file_name)
    tokens_db = Database(tokens_db_file_name)
    phrases_db = Database(phrases_db_file_name)
    with open(vocab_file_name, 'r') as f:
        vocab = json.load(f)[1:]
        pbar = ProgressBar(len(vocab))
        for i, w in enumerate(vocab):
            vocab_db[i] = w
            inverse_vocab_db[w] = i
            try:
                token, sense = w.split('|')
            except Exception:
                token = w
                sense = ''
            if token in tokens_db:
                v = tokens_db[token]
                v.append(i)
                tokens_db[token] = v
            else:
                tokens_db[token] = [i]
            if '_' in token:
                sub_tokens = token.split('_')
                x = sub_tokens[0]
                if x in phrases_db:
                    v = phrases_db[x]
                    v.append(i)
                    phrases_db[x] = v
                else:
                    phrases_db[x] = [i]
            pbar.update()
    vocab_db.close()
    tokens_db.close()
    phrases_db.close()
    frequency_db = Database(frequency_db_file_name)
    with open(frequency_file_name, 'r') as f:
        freqs = json.load(f)
    freqs = {str(inverse_vocab_db[x[0]]): str(x[1]) for x in freqs}
    inverse_vocab_db.close()
    frequency_db.update(freqs)
    frequency_db.close()
    os.remove(flag_file)
    print('Done.')


def run():
    if _is_built():
        return
    if _is_interrupted():
        _clear()
        print('Seems the previous build was interrupted. Restarting build...')
        if not _is_downloaded():
            print('Embeddings file missing. Downloading...')
            download()
    elif _is_corrupt():
        _clear()
        print('Corrupt index files. Restarting build...')
        if not _is_downloaded():
            print('Embeddings file missing. Downloading...')
            download()
    else:
        if not _is_downloaded():
            download()
        print('Seems you are running the program for the first time. Building index...')
    _build()
