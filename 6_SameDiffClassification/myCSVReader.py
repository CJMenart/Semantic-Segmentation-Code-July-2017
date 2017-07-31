import os
import fnmatch
import csv
import random
import time
import numpy as np

'''
creates a csv reader  which can pull hypercolumn training data out of a set of 
csv files and pass it out as float lists

This hardcodes the pattern in which the training data is encoded in the CSV's:
First the target value, then both hypercolumns of even length one after the other

The reader will loop over the files in a random order each time, returning more hypercolumns
for as long as you want them. It can also, optionally, reserve a few files to build a validation
set which will never be looped over, but can be retreived using get_reserve

The reader is guaranteed to reserve at LEAST reserveSize items for validation, but because
of the way the files are saved, it doesn't make sense to just have half of a file used as
validation--so it will include the whole file, possibly giving you slightly more validation
items than you asked for.

Always returns target, item A, item B, since that's the order of storage

Has two modes: One shuffles, keeping a certain number of elements in memory at all times, freely
mixing between files. The other gives you teh contents of one file every time you call read()
'''

class MyCSVReader:

	def __init__(self,dir,basename,shuffle,reserveSize=0):
		#settings
		self.minQueued = 5000
		
		# get filenames
		if not dir.endswith("\\"):
			dir = dir + "\\"		
		self.filenames = os.listdir(dir)
		self.filenames = fnmatch.filter(self.filenames,basename + "*.csv")
		self.filenames = [dir + fname for fname in self.filenames]
			
		# read up and detect the length of training features
		f = open(self.filenames[0], 'r')
		reader = csv.reader(f)
		row = next(reader)
		self.numElems = len(row)
		self.columnLen = int((self.numElems-1)/2)
		f.close()
	
		# reserve validation features
		if reserveSize:
			self.build_reserve(reserveSize)

		self.curFilenames = list(self.filenames)
		self.rows = []
		self.shuffle = shuffle
		
	def read(self,batchsize=1):
		if self.shuffle:
			return self.shuffleRead(batchsize)
		else:
			assert batchsize==1
			return self.inorderRead()
		
	def inorderRead(self):
		#returns the contents of one file
		
		if not self.filenames: #we have nothing left
			return (None,None,None,None)

		batch = []
		fname = self.filenames.pop(0)
		f = open(fname, 'r')
		reader = csv.reader(f)
		for line in reader:
			row = list(map(float,line))
			batch.append(row)
		f.close()		
		
		batch = np.array(batch)
		return(batch[:,0],batch[:,1:self.columnLen+1],batch[:,self.columnLen+1:],fname)
		
	def shuffleRead(self,batchsize):
		#returns however many training items you want
		
		while len(self.rows) < self.minQueued or len(self.rows) < batchsize: #queue up more data
			if not self.curFilenames: #when we run out, start over
				self.curFilenames = list(self.filenames)
			fname = self.curFilenames.pop(random.randint(0,len(self.curFilenames)-1))
			f = open(fname, 'r')
			reader = csv.reader(f)
			for line in reader:
				row = list(map(float,line))
				self.rows.append(row)
			f.close()		
		
		assert len(self.rows) >= batchsize		
		batch = []
		for item in range(batchsize):
			batch.append(self.rows.pop(random.randint(0,len(self.rows)-1)))
		batch = np.array(batch)
		return(batch[:,0],batch[:,1:self.columnLen+1],batch[:,self.columnLen+1:])	
		
	def build_reserve(self,reserveSize):
		self.reserve = []

		print("myCSVReader Debug: Chose the following filenames to reserve:")		
		
		'''we pop filenames out of self.filenames instead of curFilenames because these files
		are being permanently reserved.'''
		while len(self.reserve) < reserveSize:
			fname = self.filenames.pop(random.randint(0,len(self.filenames)-1))
			print(fname)
			f = open(fname, 'r')
			reader = csv.reader(f)
			for line in reader:
				row = list(map(float,line))
				self.reserve.append(row)
			f.close()
			
		self.reserve = np.array(self.reserve)
	
	def get_reserve(self):
		return self.reserve[:,0],self.reserve[:,1:self.columnLen+1],self.reserve[:,self.columnLen+1:]
		
	def test(self):
		row = "Beginning Test..."
		while row:
			print(row)
			row = self.read()
			time.sleep(0.5)
		print("Test Ended!")