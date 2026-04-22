import os
from setuptools import setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
   name='challenge',
   version='1.0',
   description='Visualize bank statements',
   packages=['challenge', 'challenge.utils'],
   install_requires=required,
)
