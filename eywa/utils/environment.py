import os


package_name = 'eywa'


def mkdir(x):
	if not os.path.isdir(x):
		os.mkdir(x)

user_path = os.path.expanduser('~')
my_dir = os.path.join(user_path, '.' + package_name)
mkdir(my_dir)
lang_dir = os.path.join(my_dir, 'lang')
mkdir(lang_dir)
lang_en_dir = os.path.join(lang_dir, 'en')
mkdir(lang_en_dir)
lang_en_embeddings_dir = os.path.join(lang_en_dir, 'embeddings')
config_file = os.path.join(my_dir, 'eywa.config')
mkdir(lang_en_embeddings_dir)
server_dir = os.path.join(my_dir, 'server')
mkdir(server_dir)
