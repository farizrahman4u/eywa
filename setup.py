from setuptools import setup
from setuptools import find_packages
import os
import sys


install_requires = ['numpy', 'dateparser', 'requests', 'numba', 'scipy']# 'mkdocs', 'mkdocs-material']

if sys.version_info[0] == 2:
      install_requires.append('pysqlite')

dependency_links = []

if os.name == 'nt':
      os.system('python -m easy_install annoy-1.8.0-py2.7-win-amd64.egg')
else:
      #dependency_links.append("git+ssh://git@github.com/farizrahman4u/annoy.git")
      install_requires.append('annoy')

setup(
      name='eywa',
      version='0.0.1',
      description='Framework for conversational AI',
      author='Fariz Rahman',
      author_email='farizrahman4u@gmail.com',
      url='https://github.com/farizrahman4u/eywa',
      license='GNU GPL v2',
      install_requires=install_requires,
      dependency_links=dependency_links,
      packages=find_packages(),
      include_package_data=True
)
