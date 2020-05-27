from abc import ABC, abstractmethod
import numpy as np

import os
import json
import numpy as np
import pandas as pd
from spacy.lang.en import English

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from gensim import corpora
import pickle
import gensim

import pyLDAvis.gensim

class modelAbClass(ABC):
	def __init__(self):
		super().__init__()
		self.dataDF = None
		self.model = None
		self.clusters = None
		self.corpus = None
		self.dictionary = None
		self.keywords = []
		self.en_stop = set(nltk.corpus.stopwords.words('english'))
		defaultSw = ["table", "-table", "-figure", "a1111111111", "author", "funder", "copyright", "org", "doi", "holder", "preprint", "peer", "reviewed", "https", "10", "org", "license", "medrxiv", "al", "et", "fig", "figure", "biorxiv", "however", "assay", "compare", "case", "horse"]
		self.en_stop.update(defaultSw)
		self.customizedStopwords = set()
		self.text_data = []
		self.lemma_stemmer_type = None
		self.token_gen_type = None
		self.parser = English()
		self.porter = PorterStemmer()
		self.lancaster = LancasterStemmer()

	@abstractmethod
	def getDF(self):
		return self.dataDF

	@abstractmethod
	def prepareText(self):
		for index, (text, abstract) in enumerate(zip(self.dataDF.text.values, self.dataDF.abstract.values), start=1):

			if self.token_gen_type not in (1,2,3,4,5,6):
				raise Exception("Invalid token_gen_type")
		
			if self.token_gen_type in (1,3,5):
				if self.token_gen_type == 1:
					textToPrepare = " ".join(abstract)
				elif self.token_gen_type == 3:
					textToPrepare = " ".join(text)
				elif self.token_gen_type == 5:
					textToPrepare = " ".join(abstract) + " " +" ".join(text)
					
				tokens = self.prepare_text_for_lda(textToPrepare)
				self.text_data.append(tokens)
				
			elif self.token_gen_type in (2,4,6):
				if self.token_gen_type == 2:
					textToPrepare = abstract
				elif self.token_gen_type == 4:
					textToPrepare = text
				elif self.token_gen_type == 6:
					textToPrepare = abstract+text
				
				for piece in textToPrepare:
					tokens = self.prepare_text_for_lda(piece)
					self.text_data.append(tokens)

			if index%1000 == 0:
					print(index, 'papers processed')


	def prepare_text_for_lda(self, text):
		tokens = self.tokenize(text[0:999999])
		tokens = [token for token in tokens if len(token) > 4]
		tokens = [self.get_lemma_stem(token) for token in tokens]
		tokens = [token for token in tokens if token not in self.en_stop]
		return tokens

	def tokenize(self, text):
		lda_tokens = []
		tokens = self.parser(text)
		for token in tokens:
			if token.orth_.isspace():
				continue
			elif token.like_url:
				lda_tokens.append('URL')
			elif token.orth_.startswith('@'):
				lda_tokens.append('SCREEN_NAME')
			else:
				lda_tokens.append(token.lower_)
		return lda_tokens

	def get_lemma_stem(self, word):
		if self.lemma_stemmer_type == "Wordnet Lemmatizer":
			lemma_stem = wn.morphy(word)
		elif self.lemma_stemmer_type == "Porter Stemmer":
			lemma_stem = self.porter.stem(word)
		elif self.lemma_stemmer_type == "Lancaster Stemmer":
			lemma_stem = self.lancaster.stem(word)
		else:
			raise Exception("Invalid lemmer/stemmer configuration")
			
		if lemma_stem is None:
			return word
		else:
			return lemma_stem
					

	@abstractmethod
	def addStopWords(self, sw):
		if sw.__class__ is not list:
			raise Exception("Stop-words provided must be a list")
		self.en_stop.update(sw)
		self.customizedStopwords.update(sw)

	@abstractmethod
	def removeStopWords(self, sw):
		if sw.__class__ is not list:
			raise Exception("Stop-words provided must be a list")
		for word in sw:
			self.en_stop.discard(word)
			self.customizedStopwords.discard(word)

	@abstractmethod
	def getStopWords(self):
		return self.en_stop

	@abstractmethod
	def loadData(self, paths, paperLimit):
		df = pd.DataFrame(columns=['title', 'abstract', 'text'])
		if paths.__class__ is not list:
			raise Exception("Paths provided must be a list")
		for path in paths:
			print('-------------------------------------------------------')
			print('Start loading data for',path,'...')
			filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
			file_number = len(filenames)
			print('A total of',file_number,'papers have been found in path')
			for index, filename in enumerate(filenames, start=1):
				with open(path+'/'+filename) as f:
					data = json.load(f)
					title = data['metadata']['title']
					try:
						abstract = [text['text'] for text in data['abstract']]
					except:
						abstract = []
					text = [text['text'] for text in data['body_text']]
				if index%1000 == 0:
					print(index, 'papers loaded out of', file_number)
				df = df.append({'title': title, 'abstract': abstract, 'text': text}, ignore_index=True)
				if paperLimit!=-1 and index >= paperLimit:
					break
			print('Finished loading data for',path,'...')
			print('-------------------------------------------------------')
		print('-------------------------------------------------------')
		print("Finished loading!")
		print('-------------------------------------------------------')
		return df


	@abstractmethod
	def filter_keyword(self, doc_df, keywords):

		print("Filtering keywords...")

		if keywords.__class__ is not list:
			raise Exception("Keywords provided must be a list")
	
		filteredByKeywords = []
		
		for keyword in keywords:
			filteredByKeywords.append(np.logical_or( 
				np.array(list(map(lambda x: any([keyword.lower() in y.lower() for y in x]), doc_df["text"]))) ,
				np.array(list(map(lambda x: any([keyword.lower() in y.lower() for y in x]), doc_df["abstract"]))) ) )
			
		doc_df["keyword"] = np.logical_and.reduce(np.array(filteredByKeywords))
		doc_df = doc_df[doc_df['keyword']].reset_index().drop(columns = ['index'])
		doc_df = doc_df.drop(columns = ['keyword'])
		
		return doc_df


	@abstractmethod
	def run(self):
		if self.dataDF is None:
			raise Exception("You must add some data with loadData() before running the model")
		self.model = None
		self.clusters = None
		self.corpus = None
		self.dictionary = None
		self.text_data = []
		self.lemma_stemmer_type = None
		self.token_gen_type = None

	@abstractmethod
	def save(self, directory):
		print("Saving model to",directory)
		try:
			os.mkdir(directory)
		except:
			print("Directory already exists!")
			return

		self.model.save(directory+'/model.gensim')
		pickle.dump(self.corpus, open(directory+'/corpus.pkl', 'wb'))
		self.dictionary.save(directory+'/dictionary.gensim')
		with open(directory+'/keywords.txt','w') as f:
			key = [", ".join(x) for x in self.keywords]
			f.write('\n'.join(key))
		with open(directory+'/stopwords.txt','w') as f:
			f.write(', '.join(self.customizedStopwords))




		