import tensorflow as tf

class SameDiffNet:
	# a siamese FC network for classifying vectors as same/different
	'''POSSIBLE TODO: 
	-Consider just messing with the layer shapes and evaluating different ones
	-Weighted distnace? Seems like there may be multiple ways to do distance layer...
	-Consider different loss functions
	-You could add an outputLen variable to the initializer, and add the possibility to go to
		longer outputs. More general. But you'll probably never need it
	-Try different normalization scheme such as L1 (may be defined outside network)
	'''
	
	def __init__(self,inputLen):
		# settings
		self.DATA_TYPE = tf.float32
		self.NORM_CONSTANT = 1e-6
		self.REGULARIZATION_WEIGHT = 1e-3
		self.MARGIN = 10.0
		self.LAYER_SIZES = [1024,1024,1024,1024,1024,1024,1024,1024,1024,512]
		self.DROPOUT_PROB = 0.50
		self.LEAK = 0.05
		
		# input
		self.x1 = tf.placeholder(self.DATA_TYPE,[None,inputLen])
		self.x2 = tf.placeholder(self.DATA_TYPE,[None,inputLen])
		self.trainMode = tf.placeholder(self.DATA_TYPE,[])
		
		# network branches
		with tf.variable_scope("siamese") as scope:
			self.branch0 = self.network_branch(inputLen,self.x1)
			scope.reuse_variables()
			self.branch1 = self.network_branch(inputLen,self.x2)
			self.regularization = self.L2_regularization(inputLen);
						
		# combination layer and loss
		self.dist = self.biased_sigmoided_euclidean_distance()
		self.target = tf.placeholder(self.DATA_TYPE,[None])
		self.out = 1 - self.dist;
		self.loss = self.cross_entropy_loss() + self.regularization
		self.accuracy = self.my_accuracy()
		
	def network_branch(self,inputLen,x):
		fc = []
		# The variance of the weights is determined using the Xavier initialization method
		weights0 = tf.get_variable("weights0", [inputLen,self.LAYER_SIZES[0]], initializer=tf.random_normal_initializer(0,2/inputLen))
		bias0 = tf.get_variable("bias0", [self.LAYER_SIZES[0]], initializer=tf.constant_initializer(0.01))
		fc.append(self.leaky_relu(tf.nn.bias_add(tf.matmul(x,weights0), bias0)))
		
		for layer in range(1,len(self.LAYER_SIZES)):
			weights = tf.get_variable("weights" + str(layer), [self.LAYER_SIZES[layer-1],self.LAYER_SIZES[layer]], initializer=tf.random_normal_initializer(0,2/self.LAYER_SIZES[layer-1]))
			bias = tf.get_variable("bias"  + str(layer), [self.LAYER_SIZES[layer]], initializer=tf.constant_initializer(0.01))
			
			# Residual skip connections!
			if layer % 2 == 0:
				fc.append(self.leaky_relu(tf.add(tf.nn.bias_add(tf.matmul(fc[-1],weights), bias), fc[-2])))
			else:
				fc.append(self.leaky_relu(tf.nn.bias_add(tf.matmul(fc[-1],weights), bias)))
		
		return fc[-1]
			
	def euclidean_distance(self):
		dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.branch0,self.branch1),2),1)+self.NORM_CONSTANT)
		return dist
		
	def biased_sigmoided_euclidean_distance(self):
		bias = tf.get_variable("biasDist",[1],initializer=tf.constant_initializer(0.0))
		dist = tf.sigmoid(tf.add(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.branch0,self.branch1),2),1)+self.NORM_CONSTANT),bias))
		return dist
	
	def biased_sigmoided_euclidean_distance_with_weights(self):
		bias = tf.get_variable("biasDist",[1],initializer=tf.constant_initializer(0.0))
		weights = tf.get_variable("weightsDist",self.LAYER_SIZES[-1],initializer=tf.random_normal_initializer(0,2/self.LAYER_SIZES[-2]))
		dist = tf.sigmoid(tf.add(tf.sqrt(tf.reduce_sum(tf.pow(tf.multiply(tf.subtract(self.branch0,self.branch1),weights),2),1)+self.NORM_CONSTANT),bias))
		return dist
	
	def cross_entropy_loss(self):
		loss = tf.reduce_sum(tf.multiply(-1.0,tf.add(tf.multiply(self.target,tf.log(self.out+self.NORM_CONSTANT)),tf.multiply(1-self.target,tf.log(1-self.out+self.NORM_CONSTANT)))))
		return loss
		
	def my_contrastive_loss(self):
		loss = tf.reduce_sum(tf.pow(tf.subtract(self.target,self.out),2))
		return loss
		
	def contrastive_loss(self):
         labelsSame = self.target
         labelsDifferent = tf.subtract(1.0, self.target)          # !labels;
         C = tf.constant(self.MARGIN, name="C")

         pos = tf.multiply(labelsSame, tf.pow(self.dist,2))
         neg = tf.multiply(labelsDifferent, tf.pow(tf.maximum(tf.subtract(C, self.dist), 0),2))
         losses = tf.add(pos, neg, name="losses")
         loss = tf.reduce_mean(losses, name="loss")
         return loss
	
	def L2_regularization(self,inputLen):
		weights0 = tf.get_variable("weights0", [inputLen,self.LAYER_SIZES[0]], initializer=tf.random_normal_initializer(0,2/inputLen))
		loss = tf.reduce_sum(tf.pow(weights0,2))
		
		for layer in range(1,len(self.LAYER_SIZES)):
			weights = tf.get_variable("weights" + str(layer), [self.LAYER_SIZES[layer-1],self.LAYER_SIZES[layer]], initializer=tf.random_normal_initializer(0,2/self.LAYER_SIZES[layer-1]))
			loss = loss + tf.reduce_sum(tf.pow(weights,2))
		
		return loss * self.REGULARIZATION_WEIGHT
	
	def hinge_similarity(self):
		return tf.maximum(0.0,tf.subtract(1.0, tf.divide(self.dist,self.MARGIN)))
		
	def untrained_cauchy_similarity(self):
		similarity = tf.divide(1.0,tf.add(1.0,self.dist))
		return similarity
		
	# doesn't work :(
	def tf_accuracy(self):
		return tf.metrics.accuracy(tf.round(self.out),self.target)
		
	def my_accuracy(self):
		return tf.reduce_mean(tf.cast(tf.equal(tf.round(self.out), self.target),tf.float32))
		
	def leaky_relu(self,x):
		return tf.maximum(x,self.LEAK*x)