import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    install_requires=["numpy", "pandas", "spacy", "nltk", "gensim", "pyLDAvis"],
    name="nuxtuNLP",
    version="0.0.1",
    author="Oscar Andres Diaz Caballero",
    author_email="oscar.diaz@nuxtu.co",
    description="NUXTU package used for Natural Language Processing. Algorithms: LDA, LSA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alegis277/nuxtuNLP/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)