from setuptools import setup
from setuptools import find_packages
import os
import sys


install_requires = ['numpy', 'dateparser', 'requests', 'scipy', 'annoy', 'responder']

if sys.version_info[0] == 2:
      install_requires.append('pysqlite')

setup(
      name='eywa',
      version='0.0.4',
      description='Framework for conversational AI',
      author='Fariz Rahman',
      author_email='farizrahman4u@gmail.com',
      url='https://github.com/farizrahman4u/eywa',
      license='GNU GPL v2',
      install_requires=install_requires,
      dependency_links=[],
      packages=find_packages(),
      include_package_data=True
)
