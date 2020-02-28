# Copyright (c) 2020 smarsu. All Rights Reserved.

"""Setup SMNet as a package.
Upload SMNet to pypi, then you can install smnet via:
    pip install smnet
Script for uploading:
```sh
(export use_cuda=true)
python setup.py sdist
twine upload dist/*
rm -r dist
rm -r smnet.egg-info
```
"""

import os
from setuptools import find_packages, setup


def config_setup(name):
  packages = find_packages()
  package_data = ['third_party/cblas/lib/libsmcb.so']

  setup(
    name = name,
    version = '0.0.0',
    packages = packages,
    package_data = {'smnet': package_data},
    install_requires = [
        'numpy',
    ],
    author = 'smarsu',
    author_email = 'smarsu@foxmail.com',
    url = 'https://github.com/smarsu/SMNet',
    zip_safe = False,
  )

print('---------------- Setup smnet ----------------')
config_setup('smnet')