from ...utils import lang_en_embeddings_dir as emb_dir
import os
import sys

frequency_file_name = os.path.join(emb_dir, 'frequency.json')
vectors_file_name = os.path.join(emb_dir, 'vectors.npy')
vocab_file_name = os.path.join(emb_dir, 'vocab.json')
vector_index_file_name = os.path.join(emb_dir, 'vectors.ann')
frequency_db_file_name = os.path.join(emb_dir, 'frequency.db')
vocab_db_file_name = os.path.join(emb_dir, 'vocab.db')
inverse_vocab_db_file_name = os.path.join(emb_dir, 'inverse_vocab.db')
phrases_db_file_name = os.path.join(emb_dir, 'phrases.db')
tokens_db_file_name = os.path.join(emb_dir, 'tokens.db')
interrupt_flag_file_name = os.path.join(emb_dir, '.interrupt_flag.tmp')
vector_size_file_name = os.path.join(emb_dir, 'vector_size.txt')
