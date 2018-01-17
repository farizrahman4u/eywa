import os
import sys
import math
import gzip
import struct
import json
import numpy as np
from .filenames import frequency_file_name, vectors_file_name, vocab_file_name
from ...utils import lang_en_embeddings_dir as emb_dir
from ...utils import ProgressBar

py3 = sys.version_info[0] == 3
if py3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen

import requests

# script to download and unzip embeddings

url = 'http://index.spacy.io/models/reddit_vectors-1.1.0/archive.gz'
meta_url = 'http://index.spacy.io/models/reddit_vectors-1.1.0/meta.json'
file_name = os.path.join(emb_dir, url.split('/')[-1])
meta_file_name = os.path.join(emb_dir + '/' + meta_url.split('/')[-1])



def _download_file(url, file_name):
    #u = urlopen(url)
    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-length'])
    '''
    if py3:
        file_size = int(u.getheader("Content-Length")[0])
    else:
        file_size = int(u.info().getheaders("Content-Length")[0])
    '''
    file_exists = False
    if os.path.isfile(file_name):
        local_file_size = os.path.getsize(file_name)
        if local_file_size == file_size:
            file_exists = True
        else:
            print("File corrupt. Downloading again.")
            os.remove(file_name)
    if not file_exists:
        factor = int(math.floor(math.log(file_size)/math.log(1024)))
        display_file_size = str(file_size / 1024 ** factor) + ['B','KB','MB','GB','TB','PB'][factor]
        print("Downloading: " + file_name + " Size: " + display_file_size)
        file_size_dl = 0
        block_sz = 8192
        f = open(file_name, 'wb')
        pbar = ProgressBar(file_size)
        for chunk in r.iter_content(chunk_size=block_sz):
            if not chunk:
                continue
            chunk_size = len(chunk)
            file_size_dl += chunk_size
            f.write(chunk)
            pbar.update(chunk_size)
            #status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            #status = status + chr(8)*(len(status)+1)
            #print(status)
        f.close()
    else:
        print("File already exists - " + file_name)
        return True


def download():
    print('Downloading meta data...')
    _download_file(meta_url, meta_file_name)
    print('Done.')

    print('Downloading embeddings...')
    _download_file(url, file_name)
    print('Done.')

    print('Extracting...')
    with open(meta_file_name, 'r') as f:
        manifest = json.load(f)['manifest']
    with gzip.open(file_name) as gzf:
        print('Extracting ' + frequency_file_name)
        file_size = manifest[0]['size']
        with open(frequency_file_name, 'w') as f:
            f.write(gzf.read(file_size))
        print('Done.')
        print('Extracting ' + vectors_file_name)
        file_size = manifest[1]['size']
        num_vectors, = struct.unpack('i', gzf.read(4))
        vector_dim, = struct.unpack('i', gzf.read(4))
        vectors = [struct.unpack('f' * vector_dim, gzf.read(4 * vector_dim)) for _ in range(num_vectors)]
        np.save(vectors_file_name, vectors)
        del vectors
        print('Done.')
        print('Extracting ' + vocab_file_name)
        file_size = manifest[2]['size']
        with open(vocab_file_name, 'w') as f:
            f.write(gzf.read(file_size))
        print('Done.')
