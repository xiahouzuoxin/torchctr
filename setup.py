import re
from setuptools import setup, find_packages

# Read the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def find_version():
    with open("torchctr/__init__.py", "r", encoding="utf-8") as rf:
        version_file = rf.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open("requirements.txt", "r", encoding="utf-8") as rf:
    requirements = [line.strip() for line in rf.readlines()]

setup(
    name='torchctr',
    version=find_version(),
    author='zuoxin.xiahou',
    author_email='xiahouzuoxin@163.com',
    description='A small pytorch implementation for ctr prediction in recommendation system for small companies',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiahouzuoxin/torchctr",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'all': [
            'accelerate>=1.2.1',
            'scikit-learn>=1.5.1',
            'tensorboard',
            'uvicorn',
            'fastapi',
            'pydantic'
        ]
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)