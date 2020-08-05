from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import os


class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/' 
	#fnTrain = '/content/' # for colab, copy the data from drive.
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'


def train(model, loader, datablock_size):
	"train NN"
	model.trainBatch(loader, FilePaths.fnAccuracy, datablock_size)


def infer(model, batch):
	"recognize text in image provided by file path"
	if DecoderType == DecoderType.WordBeamSearch:
		recognized = model.inferBatch(batch)
		print('result is ')
		print(recognized)
		return recognized
	else:
		(recognized, probability) = model.inferBatch(batch)
		print(recognized)
		print('Probability:', probability)
		return recognized


def main():
	"main function"
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	parser.add_argument('--batch_size', help='batch size? default is 100', type=int, default=100)
	parser.add_argument('--block_size', help='datablock size? default is 10,000. decrease if RAM is lower than 16gigs', \
		type=int, default=10000)

	args = parser.parse_args()

	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	batch_size = args.batch_size
	datablock_size = args.block_size

	# train on IAM dataset	
	if args.train:
		# load training data, create TF model
		print('Setting up loader')
		loader = DataLoader(FilePaths.fnTrain, Model.imgSize, Model.maxTextLen)

		# save characters of model for inference mode
		print('Writing charlist')
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		print('Writing words')
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.words))

		# execute training
		if args.train:
			print('Setting up model')
			model = Model(loader.charList, decoderType, batch_size)
			# train(model, loader)
			train(model, loader, datablock_size)

	# infer text on test image
	else:
		file = open(FilePaths.fnInfer + 'preds.txt', 'w')
		model = Model(open(FilePaths.fnCharList).read(), decoderType, batch_size=100)
		imgs = []
		for filename in os.listdir(FilePaths.fnInfer):
			img = FilePaths.fnInfer + filename
			imgs.append(preprocess(cv2.imread(img, cv2.IMREAD_GRAYSCALE), Model.imgSize))
		size = len(imgs)
		dummy_txts = [None for i in range(size)]
		batch = Batch(dummy_txts, imgs)
		preds = infer(model, batch)
		for word in preds:
			file.write(word+' ')
		file.close()


if __name__ == '__main__':
	main()