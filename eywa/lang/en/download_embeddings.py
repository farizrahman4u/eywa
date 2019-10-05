import requests
import os
import sys
import math
import tarfile
import struct
import json
import shutil
import numpy as np
from .filenames import frequency_file_name, vectors_file_name, vocab_file_name
from .filenames import interrupt_flag_file_name as flag_file
from ...utils import lang_en_embeddings_dir as emb_dir
from ...utils import ProgressBar

py3 = sys.version_info[0] == 3
if py3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen


# script to download and unzip embeddings

url = 'https://github.com/explosion/sense2vec/releases/download/' + \
      'v1.0.0a0/reddit_vectors-1.1.0.tar.gz'
file_name = url.split('/')[-1]
dir_name = os.path.join(emb_dir, file_name[:-7])
file_name = os.path.join(emb_dir, file_name)


def _download_file(url, file_name):
    # u = urlopen(url)
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
        factor = int(math.floor(math.log(file_size) / math.log(1024)))
        display_file_size = str(file_size / 1024 ** factor) + \
            ['B', 'KB', 'MB', 'GB', 'TB', 'PB'][factor]
        print("Source: " + url)
        print("Destination " + file_name)
        print("Size: " + display_file_size)
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
            # status = r"%10d  [%3.2f%%]" %
            #                       (file_size_dl, file_size_dl * 100. / file_size)
            # status = status + chr(8)*(len(status)+1)
            # print(status)
        f.close()
    else:
        print("File already exists - " + file_name)
        return True


def download():
    print('Downloading embeddings...')
    _download_file(url, file_name)
    print('Done.')

    print('Extracting...')
    with tarfile.open(file_name, 'r:gz') as tf:
        tf.extractall(emb_dir)
    print('Done.')
    os.remove(os.path.join(dir_name, 'meta.json'))
    files = os.listdir(dir_name)
    for f in files:
        src_f = os.path.join(dir_name, f)
        dest_f = os.path.join(emb_dir, f)
        shutil.move(src_f, dest_f)
    os.rename(os.path.join(emb_dir, 'strings.json'), vocab_file_name)
    os.rename(os.path.join(emb_dir, 'freqs.json'), frequency_file_name)
    print('Converting embeddings...')
    with open(os.path.join(emb_dir, 'vectors.bin'), 'rb') as f:
        num_vectors = struct.unpack('i', f.read(4))[0]
        vector_dim = struct.unpack('i', f.read(4))[0]
        vectors = [
            struct.unpack(
                'f' *
                vector_dim,
                f.read(
                    4 *
                    vector_dim)) for _ in range(num_vectors)]
        np.save(vectors_file_name, vectors)
        del vectors
    print('Done.')
