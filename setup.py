#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Rusty Nail",
    author_email='rustemgal@gmail.com',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="Prepare training data, train models and make deployments very quickly",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='finegrained',
    name='finegrained',
    packages=find_packages(include=['finegrained', 'finegrained.*']),
    # test_suite='tests',
    # tests_require=test_requirements,
    url='https://github.com/rusteam/finegrained',
    version='0.1.0',
    entry_points={
        "console_scripts": ['finegrained=finegrained.cli:main']
    },
    zip_safe=False,
)
