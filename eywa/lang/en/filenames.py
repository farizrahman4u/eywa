from ...utils import lang_en_embeddings_dir as emb_dir
import os
import sys

py_version = '_py' + sys.version[0]

frequency_file_name = os.path.join(emb_dir, 'frequency.json')
vectors_file_name = os.path.join(emb_dir, 'vectors.npy')
vocab_file_name = os.path.join(emb_dir, 'vocab.json')
vector_index_file_name = os.path.join(emb_dir, 'vectors.ann')
frequency_db_file_name = os.path.join(emb_dir, 'frequency{}.db'.format(py_version))
vocab_db_file_name = os.path.join(emb_dir, 'vocab{}.db'.format(py_version))
inverse_vocab_db_file_name = os.path.join(emb_dir, 'inverse_vocab{}.db'.format(py_version))
phrases_db_file_name = os.path.join(emb_dir, 'phrases{}.db'.format(py_version))
tokens_db_file_name = os.path.join(emb_dir, 'tokens{}.db'.format(py_version))
interrupt_flag_file_name = os.path.join(emb_dir, '.interrupt_flag{}.tmp'.format(py_version))
vector_size_file_name = os.path.join(emb_dir, 'vector_size.txt')