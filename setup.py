#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'statsmodels', 'matplotlib']

test_requirements = []

setup(
    author="Esfira Babajanyan",
    author_email='esfira.babajanyan2002@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Implementation of the Bass Model for innovation diffusion",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bassmodel',
    name='innovationdiffusion',
    packages=find_packages(include=['innovationdiffusion', 'innovationdiffusion.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/esfira02/bassmodel',
    version='0.0.1',
    zip_safe=False,
)
