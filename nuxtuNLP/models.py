from nuxtuNLP.abstractClasses import model

import os
import json
import numpy as np
import pandas as pd
from spacy.lang.en import English

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from gensim import corpora
import pickle
import gensim

import pyLDAvis.gensim


class LDA(model.modelAbClass):
	def __init__(self):
		super().__init__()
		self.passes = None
		self.iterations = None

	def addStopWords(self, sw):
		super().addStopWords(sw)

	def removeStopWords(self, sw):
		super().removeStopWords(sw)

	def getStopWords(self):
		return super().getStopWords()

	def loadData(self, keywords, paperLimit, paths):
		self.keywords.append(keywords)
		df = super().loadData(paths, paperLimit)
		df = self.filter_keyword(df, keywords)

		if self.dataDF is None:
			self.dataDF = df
		else:
			self.dataDF = pd.concat([self.dataDF, df], ignore_index=True)

	def getDF(self):
		return super().getDF()

	def filter_keyword(self, doc_df, keywords):
		return super().filter_keyword(doc_df, keywords)

	def prepareText(self):
		super().prepareText()

	def run(self, clusters, token_gen_type, lemma_stemmer_type, passes, iterations):
		super().run()

		self.token_gen_type = token_gen_type
		self.lemma_stemmer_type = lemma_stemmer_type

		print("Processing text DataFrame...")

		self.prepareText()

		print("Training the LDA model...")

		self.clusters = clusters
		self.dictionary = corpora.Dictionary(self.text_data)
		self.corpus = [self.dictionary.doc2bow(text) for text in self.text_data]
		self.passes = passes
		self.iterations = iterations

		self.model = gensim.models.LdaMulticore(self.corpus, num_topics = self.clusters, id2word = self.dictionary, passes = self.passes, iterations = self.iterations)

		topics = self.model.print_topics(num_words=4)
		for topic in topics:
			print(topic)

	def save(self, directory, LDAvis = True):
		super().save(directory)
		if LDAvis:
			lda_display = pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary, sort_topics=False)
			pyLDAvis.save_html(lda_display, directory+'/lda.html')





class LSA(model.modelAbClass):
	def __init__(self):
		super().__init__()

	def addStopWords(self, sw):
		super().addStopWords(sw)

	def removeStopWords(self, sw):
		super().removeStopWords(sw)

	def getStopWords(self):
		return super().getStopWords()

	def loadData(self, keywords, paperLimit, paths):
		self.keywords.append(keywords)
		df = super().loadData(paths, paperLimit)
		df = self.filter_keyword(df, keywords)

		if self.dataDF is None:
			self.dataDF = df
		else:
			self.dataDF = pd.concat([self.dataDF, df], ignore_index=True)

	def getDF(self):
		return super().getDF()

	def filter_keyword(self, doc_df, keywords):
		return super().filter_keyword(doc_df, keywords)

	def prepareText(self):
		super().prepareText()

	def run(self, clusters, token_gen_type, lemma_stemmer_type):
		super().run()

		self.token_gen_type = token_gen_type
		self.lemma_stemmer_type = lemma_stemmer_type

		print("Processing text DataFrame...")

		self.prepareText()

		print("Training the LSA model...")

		self.clusters = clusters
		self.dictionary = corpora.Dictionary(self.text_data)
		self.corpus = [self.dictionary.doc2bow(text) for text in self.text_data]

		self.model = gensim.models.LsiModel(self.corpus, num_topics = self.clusters, id2word = self.dictionary)
		topics = self.model.print_topics(num_topics = self.clusters, num_words=4)
		for topic in topics:
			print(topic)

	def save(self, directory):
		super().save(directory)
