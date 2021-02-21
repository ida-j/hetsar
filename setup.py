# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='hetsar',
    version='0.1',
    description='Package for estimating the Heterogeneous SAR model (HSAR)',
#     long_description=readme,
    author='Ida Johnsson',
    author_email='ida.b.johnsson@gmail.com',
    url='https://github.com/ida-j/hetsar',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

