from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess


class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

	def __init__(self, filePath, imgSize, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'

		self.currIdx = 0
		self.imgSize = imgSize
		self.samples = []
	
		f=open(filePath+'words.txt')
		chars = set()
		count = 0
		bad_samples = []
		bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
		for line in f:
			# ignore comment line
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9
			
			# filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

			# GT text are columns starting at 9
			gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
			chars = chars.union(set(list(gtText)))

			# check if image is not empty
			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue

			# put sample into list
			self.samples.append(Sample(gtText, fileName))
			count += 1
			if count % 20000 == 0:
				print(str(count) + ' files checked')


		# some images in the IAM dataset are known to be damaged, don't show warning for them
		if set(bad_samples) != set(bad_samples_reference):
			print("Warning, damaged images found:", bad_samples)
			print("Damaged images expected:", bad_samples_reference)

		splitIdx = int(0.95 * len(self.samples))
		random.shuffle(self.samples)
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		# put words into lists
		self.words = [x.gtText for x in self.samples]

		# list of all chars in dataset
		self.charList = sorted(list(chars))

		# total number of elements
		self.total_num_elements = len(self.samples)

		# total number of elements in train sample
		self.total_train_elements = len(self.trainSamples)


	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text

	
	def trainSet(self):
		self.samples = self.trainSamples
		self.dataAugmentation = True
	

	def validSet(self):
		self.samples = self.validationSamples
		self.dataAugmentation = False
	

	def load_generator(self, datablock_size):
		"python generator to get datablocks"
		curr_block = 1
		while True:
			curr_block += 1
			random.shuffle(self.samples)
			trainSamples_sub = self.samples[:datablock_size]
			gtTexts = [trainSamples_sub[i].gtText for i in range(datablock_size)]
			imgs = []
			for i in range(datablock_size):
				imgs.append(preprocess(cv2.imread(trainSamples_sub[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation))
			assert len(gtTexts) == len(imgs)
			assert len(imgs) == datablock_size
			yield Batch(gtTexts, imgs)