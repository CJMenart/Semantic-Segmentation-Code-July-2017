''' Training a Siamese network to identify hypercolumns as belonging to 
either the same class or different classes .
Christopher Menart, 7/5/17'''

import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sameDiffNet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import myCSVReader
import random


def sameDiffTesting(checkpointDir,dataDir,dataBasename):
	
	reader = myCSVReader.MyCSVReader(dataDir,dataBasename,False)
	tf.reset_default_graph()
	sess = tf.InteractiveSession()
	network = sameDiffNet.SameDiffNet(reader.columnLen)
	
	bestAcc = tf.Variable(tf.zeros([1]),name="bestAcc")
	saver = tf.train.Saver()
	latestModel = tf.train.latest_checkpoint(checkpointDir)
	assert latestModel
	saver.restore(sess,latestModel)
	print('Current validation accuracy: %.3f' % (np.mean(sess.run([bestAcc]))))
	
	_,test1,test2,fname = reader.read()
	while fname:

		embed = network.out.eval({
				network.x1:test1,
				network.x2:test2,
				network.trainMode:0.0})
		embed.tofile(fname[:-4] + '_results.txt',sep=",",format="%.3f")
		
		print("Completed File %s" % fname)
	
		_,test1,test2,fname = reader.read()
	
	sess.close()
	print("Done.")

def sameDiffTraining(checkpointDir,dataDir,dataBasename,numExamples):
	#Settings
	#numVal = round(numExamples/20)
	numIter = round(numExamples/10)
	batchSize = 500

	#Open training data and randomly pull out some as a validation set
	random.seed(1492) #ensures that the same validation set will be randomly selected each time we boot
	reader = myCSVReader.MyCSVReader(dataDir,dataBasename,True)
	valTarget, val1, val2 = reader.get_reserve()
	
	tf.reset_default_graph()
	#with tf.device('/gpu:0'):   #manual placement unnecessary, tensorflow handles
	sess = tf.InteractiveSession()
	network = sameDiffNet.SameDiffNet(reader.columnLen)
	optimizer = tf.train.AdamOptimizer()
	train = optimizer.minimize(network.loss)
	gradients = optimizer.compute_gradients(network.loss)
	applyGrad = optimizer.apply_gradients(gradients)
	bestAcc = tf.Variable(tf.zeros([1]),name="bestAcc")

	saver = tf.train.Saver()
	latestModel = tf.train.latest_checkpoint(checkpointDir)
	if latestModel:
		saver.restore(sess,latestModel)
		start = int(latestModel.split('-')[-1])+1
		print('Starting from iteration %d. Current validation accuracy: %.3f' % (start, np.mean(sess.run([bestAcc]))))
	else:
		start = 0
		tf.global_variables_initializer().run()
	modelName = checkpointDir+'SiameseNet'
		
	#debugging
	'''
	print("Trainable Variables:")
	print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	'''

	for iter in range(start,numIter):
		''' I don't think this actually does anything
		if iter == 10000:
			batchSize = 1000
			print("Increasing batch size to %d" % batchSize)
		if iter == 15000:
			batchSize = 2000
			print("Increasing batch size to %d" % batchSize)
		'''
		
		target,train1,train2 = reader.read(batchSize)
		
		# you can run the optimization in two steps or one
		'''
		grad = sess.run([gradients], feed_dict={
						network.x1:train1,
						network.x2:train2,
						network.target: target})
		'''
		'''
		sess.run([applyGrad])
		'''
		
		_, loss = sess.run([train,network.loss], feed_dict={
						network.x1:train1,
						network.x2:train2,
						network.target: target,
						network.trainMode: 1.0})
			
		avgLoss = np.mean(loss)
		if np.isnan(avgLoss):
			print('Model diverged with loss = NaN')
			quit()

		acc = sess.run([network.accuracy], feed_dict={
				network.x1:val1,
				network.x2:val2,
				network.target: valTarget,
				network.trainMode: 0.0})
			
		acc = np.mean(acc)
		if iter % 5 == 0:
			print ('step %d: loss %.3f' % (iter, avgLoss))
			print ('step %d: val accuracy %.3f' % (iter, acc))
			
		#optional debugging stuff
		'''
		print("Gradients:")
		print(grad)
		'''
		'''
		print("First-Layer Weights:")
		print(sess.run(network.branchWeights["weights"][0]))
		'''
		
		'''
		It turns out that we cannot make periodic saves and still keep 'the best network so far'
		without going to some complicated measures. Since I can also keep the best network by just commenting
		this out that's what I'm going to do for now.
		if iter % 100 == 0 and iter > 0:
			saver.save(sess, modelName,global_step = iter)
		'''
			
		if acc > np.mean(sess.run([bestAcc])):
			sess.run([bestAcc.assign([acc])])
			saver.save(sess, modelName+"Best",global_step = iter)
		
	sess.close()
	print("Done.")