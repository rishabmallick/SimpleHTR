from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os
import datetime
import time
import editdistance


class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


# Functional API instead of above implementation
# https://github.com/tensorflow/tensorflow/issues/31647#issuecomment-531897680
def htrmodel_functional(height, width, depth, batch_size, is_train, charList):
	def cnn(inputs, is_train, kernelVals, featureVals, strideVals, poolVals, numLayers):
		x = inputs
		for i in range(numLayers):
			x = tf.keras.layers.Conv2D(filters=featureVals[i+1], kernel_size=kernelVals[i],\
				padding='same', strides=1)(x)
			x = tf.keras.layers.BatchNormalization()(inputs=x, training=is_train)
			x = tf.keras.layers.ReLU()(x)
			x = tf.nn.max_pool(input=x, ksize=(1, poolVals[i][0], poolVals[i][1], 1),\
				strides=(1, strideVals[i][0], strideVals[i][1], 1), padding='VALID')
		# BxTx1xF
		output = x
		return output

	def rnn(inputs, charList, numHidden=256):
		cells = [tf.keras.layers.LSTMCell(units=numHidden) for _ in range(2)]
		stacked = tf.keras.layers.StackedRNNCells(cells)
		lstm_layer = tf.keras.layers.RNN(stacked, return_sequences=True)
		x = inputs
		# BxTx1xF -> BxTxF
		x = tf.squeeze(x, axis=[2])
		# BxTxF -> BxTx2H
		x = tf.keras.layers.Bidirectional(layer=lstm_layer, merge_mode='concat', dtype=x.dtype)(x)
		# BxTx2H -> BxTxC
		#x = tf.keras.layers.Dense(len(charList) + 1, activation='relu')(x)
		# BxTx2H -> BxTx1X2H
		x = tf.expand_dims(x, 2)
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC, equivalent of atrous_conv2d with rate 1
		x = tf.keras.layers.Conv2D(filters=len(charList) + 1, kernel_size=1,\
			padding='same', strides=1)(x)
		# BxTx1xC -> BxTxC
		x = tf.squeeze(x, axis=[2])
		output = x
		return output

	inputShape = (height, width, depth)
	inputs = tf.keras.layers.Input(shape=inputShape, batch_size=batch_size)
	is_train = is_train
	kernelVals = [5, 5, 3, 3, 3]
	featureVals = [1, 32, 64, 128, 128, 256]
	strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
	numLayers = len(strideVals)
	charList = charList

	# CRNN aproach
	x = cnn(inputs, is_train, kernelVals, featureVals, strideVals, poolVals, numLayers)
	x = rnn(x, charList)

	# make the model 
	model = tf.keras.Model(inputs, x, name="htr_model")
	return model
			

class Model:
	''' Final pipeline model '''
	# model constants
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, batch_size=100):
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			# Restrict TensorFlow to only allocate 6 GB of memory on the first GPU
			# to deal with cudNN failed errors
			try:
				tf.config.experimental.set_virtual_device_configuration(
					gpus[0],
					[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)])
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
			except RuntimeError as e:
				# Virtual devices must be set before GPUs have been initialized
				print(e)
				
		self.charList = charList
		self.decoderType = decoderType
		self.batch_size = batch_size

		# decay learning rate
		# decay_steps=18000 was fine for upto epoch 20
		# bad decay_steps could result in 'No valid path found' for ctc_loss mid-training
		# https://stackoverflow.com/a/51853325
		self.rate = tf.keras.optimizers.schedules.PolynomialDecay(
    		initial_learning_rate=0.001,
  		  	decay_steps=10000, 
    		end_learning_rate=0.0001,
    		power=1.0,
    		cycle=False,
    		name=None,
		)
		self.model = htrmodel_functional(128, 32, 1, batch_size=self.batch_size, is_train=True, \
			charList=self.charList)
		self.opt = tf.keras.optimizers.RMSprop(learning_rate=self.rate)
		self.model_loss = self.loss_fn(logit_length=[Model.maxTextLen] * self.batch_size)
		tf.keras.utils.plot_model(self.model, 'model.png', show_shapes=True, rankdir='TB')


	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor 
			decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	# https://stackoverflow.com/a/55057315
	def custom_loss(self, y_true, y_pred, logit_length):
		loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, \
			logit_length=logit_length, label_length=None, blank_index=-1, logits_time_major=False))
		return loss

	def loss_fn(self, logit_length):
		def loss_op(y_true, y_pred):
			return self.custom_loss(y_true, y_pred, logit_length)
		return loss_op


	def get_tf_dataset(self, dataset):
		datablock_size = len(dataset.imgs)
		assert len(dataset.imgs) == len(dataset.gtTexts)
		allImgs = np.asarray(dataset.imgs)
		allImgs = tf.expand_dims(input=allImgs, axis=3)

		inputImgs = allImgs[:datablock_size]
		inputTxts = self.toSparse(dataset.gtTexts[:datablock_size])
		labels = tf.sparse.SparseTensor(inputTxts[0], inputTxts[1], inputTxts[2])
		tr_dataset = tf.data.Dataset.from_tensor_slices((inputImgs, labels))
		tr_dataset = tr_dataset.shuffle(buffer_size=len(inputImgs))
		tr_dataset = tr_dataset.batch(self.batch_size)
		return tr_dataset


	def get_val_tf_dataset(self, dataset):
		datablock_size = len(dataset.imgs)
		assert len(dataset.imgs) == len(dataset.gtTexts)
		allImgs = np.asarray(dataset.imgs)
		allImgs = tf.expand_dims(input=allImgs, axis=3)		

		validImgs = allImgs[:datablock_size]
		validTxts = dataset.gtTexts[:datablock_size]

		val_dataset = tf.data.Dataset.from_tensor_slices((validImgs, validTxts))
		# val_dataset = tf.data.Dataset.from_tensor_slices(validImgs)
		val_dataset = val_dataset.batch(self.batch_size)
		return val_dataset


	@tf.function
	def ctc_decoder(self, rnnOutput, seqLen):
		rnnOutput = tf.transpose(rnnOutput, [1, 0, 2])
		ctcOutput = tf.nn.ctc_greedy_decoder(inputs=rnnOutput, sequence_length=seqLen)
		return ctcOutput


	def validation(self, val_dataset):
		numCharErr = 0
		numCharTotal = 0
		numWordOK = 0
		numWordTotal = 0
		#print('Ground truth -> Recognized')
		for img_batch, txt_batch in val_dataset:
			#rnnOutput = self.test_step(img_batch)
			rnnOutput = self.test_step(img_batch)
			seqLen = [Model.maxTextLen] * self.batch_size
			ctcOutput = self.ctc_decoder(rnnOutput, seqLen)
			recognized = self.decoderOutputToText(ctcOutput, self.batch_size)
			for i in range(len(recognized)):
				txt = txt_batch[i].numpy()
				txt = txt.decode("utf-8")
				numWordOK += 1 if txt == recognized[i] else 0
				numWordTotal += 1
				dist = editdistance.eval(recognized[i], txt)
				numCharErr += dist
				numCharTotal += len(txt)
				#print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + txt + '"', '->', '"' + recognized[i] + '"')

		# print validation result
		charErrorRate = numCharErr / numCharTotal
		wordAccuracy = numWordOK / numWordTotal
		return charErrorRate, wordAccuracy


	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			logits = self.model(x)
			loss_value = self.model_loss(y, logits)
		grads = tape.gradient(loss_value, self.model.trainable_weights)
		self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
		return loss_value
	
	# https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-593001572
	def train_step_wrapper(self, nn_model, loss_func, optimiser):
		'''
		# code snippet. incase it's useful
		try:
			loss_value = train_step(x_batch_train, y_batch_train)
		except(ValueError, UnboundLocalError):
			train_one_batch = train_step_wrapper(self.model, self.model_loss, self.opt)
			oss_value = train_one_batch(x_batch_train, y_batch_train)
		'''
		@tf.function
		def train_one_batch(x_train, y_train):
			with tf.GradientTape() as tape:
				y_pred = nn_model(x_train)
				loss = loss_func(y_train, y_pred)
			grads = tape.gradient(loss, nn_model.trainable_variables)
			optimiser.apply_gradients(zip(grads, nn_model.trainable_variables))
			return loss
		return train_one_batch


	@tf.function
	def test_step(self, x):
		logits = self.model(x)
		return logits


	def trainBatch(self, loader, path_acc, datablock_size=10000):
		print('batch size is ', self.batch_size)

		total_block = int(loader.total_train_elements/datablock_size)
		print('Total %d blocks of %d elements' %(total_block, datablock_size))

		checkpoint_dir = '../model/weights/'
		checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=self.model, optimizer=self.opt)
		# step here is epoch, followed guide from
		# https://www.tensorflow.org/guide/checkpoint
		manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

		# resuming from previous training session, if any
		checkpoint.restore(manager.latest_checkpoint)
		ckpt_epoch = 0
		if manager.latest_checkpoint:
			ckpt_epoch = int(checkpoint.step)
			print("Restored from {}".format(manager.latest_checkpoint))
			print('Resuming training from epoch ', ckpt_epoch)
			print('Saving the checkpoints in', checkpoint_dir)

		print('getting validation data ready...')
		loader.validSet()
		val_generator = loader.load_generator(datablock_size=5000) # fixed. 5% of total 100k.
		valBatch = next(val_generator)
		val_dataset = self.get_val_tf_dataset(valBatch)
		loader.trainSet()
		data_generator = loader.load_generator(datablock_size=datablock_size)

		patience = 5
		best_loss = 10000.0
		early_epoch = -1
		epochs = 50 - ckpt_epoch

		for epoch in range(epochs):
			start_time = time.time()
			curr_block = 0
			bool_save_model = False
			print("\nStart of epoch %d" % (epoch+ckpt_epoch,))
			while True:
				curr_block += 1
				print('Loading datablock no. '+str(curr_block)+' out of '+str(total_block))
				dataset = next(data_generator)
				tr_dataset = self.get_tf_dataset(dataset)
				if curr_block == total_block:
					print('End of training for epoch %d' %(epoch+ckpt_epoch))
					break
				for step, (x_batch_train, y_batch_train) in enumerate(tr_dataset):
					loss_value = self.train_step(x_batch_train, y_batch_train)
					if step % 80 == 0:
						print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
						print("Seen so far: %d samples of block: %d" % ((step + 1) * self.batch_size, curr_block))
			
			print('Validating...')
			charErrorRate, wordAccuracy = self.validation(val_dataset)
			print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
			if best_loss > charErrorRate:
				print('improving...')
				best_loss = charErrorRate
				bool_save_model = True
				early_epoch = 0
			else:
				early_epoch += 1
				bool_save_model = False
				# early stopping
				if early_epoch > patience:
					print('ending training prematurely due to no improvement')
					break

			# saving only after going through every block in an epoch
			if bool_save_model is True:
				# https://github.com/tensorflow/tensorflow/issues/31057#issuecomment-523262141
				# saving the model
				# https://github.com/tensorflow/tensorflow/issues/37973#issuecomment-605448707
				# self.model.save_weights('../model/weights/')
				# checkpoint.save(file_prefix=checkpoint_prefix)
				save_path = manager.save()
				checkpoint.step.assign_add(1)
				print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.step - 1), save_path))
				open(path_acc, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
			end_time = time.time()
			print('Training epoch ' + str(epoch+ckpt_epoch) + ' took ' + str(end_time-start_time) + ' seconds.')

		print('training complete')

		# save the model from best previous checkpoint
		checkpoint.restore(manager.latest_checkpoint)

		# sanity check
		print('Sanity check. Validating again after restoring...')
		charErrorRate, wordAccuracy = self.validation(val_dataset)
		print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))

		ckpt_epoch = 0
		if manager.latest_checkpoint:
			ckpt_epoch = int(checkpoint.step)
			print("Restored from {}".format(manager.latest_checkpoint))
		print('saving model in SavedModel format')
		save_path = '../model/savedmodel/'
		tf.saved_model.save(self.model, save_path)
		

	def inferBatch(self, batch):
		images = batch.imgs
		# loading model
		path = '../model/savedmodel/'
		imported_model = tf.saved_model.load(path)
		print('Succesfully loaded saved model')
		numBatchElements = len(images)
		allImgs = np.asarray(images)
		allImgs = tf.expand_dims(input=allImgs, axis=3)
		allImgs = np.float32(allImgs) # cuz model configured with float32 inputs 
		print('getting preds...')
		rnnOutput = imported_model(allImgs)
		# rnnOutput = model(batch.imgs) # Not best thing. See below
		# search .predict here https://www.tensorflow.org/guide/data
		# https://keras.io/api/models/model_training_apis/
		seqLen = [Model.maxTextLen] * numBatchElements
		if self.decoderType == DecoderType.BestPath:
			ctcOutput = tf.nn.ctc_greedy_decoder(inputs=rnnOutput, sequence_length=seqLen)

		# results are similar to above approach
		elif self.decoderType == DecoderType.BeamSearch:
			ctcOutput = tf.nn.ctc_beam_search_decoder(inputs=rnnOutput, \
				sequence_length=seqLen, beam_width=50, merge_repeated=False)

		# haven't added this feature yet.
		#elif self.decoderType == DecoderType.WordBeamSearch:

		texts = self.decoderOutputToText(ctcOutput, numBatchElements)
		sparse = self.toSparse(texts)
		sparse_tensor = tf.sparse.SparseTensor(sparse[0], sparse[1], sparse[2])
		loss = tf.nn.ctc_loss(labels=sparse_tensor, logits=rnnOutput, \
			logit_length=seqLen, label_length=None, blank_index=-1)
		probs = np.exp(-loss)
		return texts, probs