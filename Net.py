import tensorflow as tf
import utils
import os

class Net:
	"""
	X: 		input np.array representing images 
			in the shape (num_images, image_length ** 2)

	Y_TRUE: labels for each x, np.array, (num_images, num_classes)

	ARGS: 	if you want to change constants from their default setting,
			specify in args
	Reference: [1]
	"""

	def __init__(self, x=None, y_true=None, train=True, \
				load=True, name="trained_net", args={}):
		#Constants
		self.FLAGS = {
			"filter_size1" : 5, 
			"num_filters1" : 16, 
			"image_length" : 128, 
			"num_channels" : 1, 
			"num_classes" : 2, 
			"filter_size2" : 5, 
			"num_filters2" : 36, 
			"fc_size" : 128, 
			"learning_rate": 1e-4, 
			"train_batch_size" : 64, 
			"eval_batch_size" : 64, 
			"num_epochs" : 1
		}

		for key, val in args.items():
			if key in self.FLAGS:
				self.FLAGS[key] = val

		self.sess = tf.Session()
		self.build_graph()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

		checkpoint_folder = f"nets/{name}"
		checkpoint_file = f"nets/{name}/{name}"

		if load and utils.path_exists(f"{checkpoint_folder}/checkpoint"):
			self.load_net(checkpoint_file, checkpoint_folder)
		elif load:
			print("No Checkpoint available to load")

		if train and x is not None and y_true is not None:
			assert x.shape[0] == y_true.shape[0], \
				f"First dim of x {x.shape} must be \
				same size as y_true {y_true.shape}"
			if not utils.path_exists(checkpoint_folder):
				os.makedirs(checkpoint_folder)
			self.train_net(checkpoint_file, checkpoint_folder, \
							x, y_true, self.FLAGS["num_epochs"])
	@utils.timeit
	def load_net(self, checkpoint_file, checkpoint_folder):
		self.saver.restore(self.sess, checkpoint_file)

		arg_file = f"{checkpoint_folder}/FLAGS.args"
		if utils.path_exists(arg_file):
			with open(arg_file) as f:
				self.FLAGS = eval(f.readline())
		else:
			print("\nUnable to restore flags\n")

		print("\nCheckpoint Restored\n")

	@utils.timeit
	def save_net(self, checkpoint_file, checkpoint_folder):
		self.saver.save(self.sess, checkpoint_file)

		arg_file = f"{checkpoint_folder}/FLAGS.args"

		with open(arg_file, 'w+') as f:
			f.write(str(self.FLAGS))


		print("\nCheckpoint Saved\n")


	@utils.timeit
	def train_net(self, checkpoint_file, checkpoint_folder, \
					x, y_true, num_epochs=1):
		batch_size = self.FLAGS["train_batch_size"]
		num_batches = len(x) // batch_size
		for batch_num in range(num_batches * num_epochs):
			curr_sample = ((batch_num % num_batches) * batch_size)
			next_sample = min(curr_sample+batch_size, num_batches*batch_size)
			
			x_batch = x[curr_sample : next_sample]

			y_true_batch = y_true[curr_sample : next_sample]

			feed_dict_train = {self.x : x_batch, self.y_true : y_true_batch}

			#forward pass
			self.sess.run(self.optimizer, feed_dict=feed_dict_train)

			#train status
			if batch_num % num_batches == num_batches - 1:

				print("\n\nCompleted an epoch")

				acc = self.sess.run(self.accuracy, feed_dict=feed_dict_train)

				print(f"Training Accuracy: {acc}\n\n")
		
		self.save_net(checkpoint_file, checkpoint_folder)

	@utils.timeit
	def classify(self, x):

		feed_dict_test = {self.x: x}

		prediction = self.sess.run(self.y_pred_cls, feed_dict= feed_dict_test)

		return prediction

	def close(self):
		self.sess.close()

	def build_graph(self):

		
		img_size = self.FLAGS["image_length"]
		num_channels = self.FLAGS["num_channels"]
		num_classes = self.FLAGS["num_classes"]

		#input
		self.x = tf.placeholder(
			tf.float32, 
			shape=[None, img_size**2, num_channels], 
			name='x'
		)

		x_image = tf.reshape(self.x, [-1, img_size, img_size, num_channels])

		self.y_true = tf.placeholder(
			tf.float32, 
			shape=[None, num_classes], 
			name='y_true'
		)

		y_true_cls = tf.argmax(self.y_true, axis=1)



		#conv1
		conv1, weights_c1 = self._convolutional_layer(
			x_image, 
			self.FLAGS["num_channels"], 
			self.FLAGS["filter_size1"], 
			self.FLAGS["num_filters1"]
		)

		#pool1
		pool1 = self._max_pooling_layer(conv1)

		#relu1
		relu1 = self._relu_layer(pool1)

		#conv2
		conv2, weights_c2 = self._convolutional_layer(
			relu1,
			self.FLAGS["num_filters1"],
			self.FLAGS["filter_size2"],
			self.FLAGS["num_filters2"]
		)

		#pool2
		pool2 = self._max_pooling_layer(conv2)

		#relu2
		relu2 = self._relu_layer(pool2)

		#flatten from 4 to 2 dimensions
		flat, num_features = self._flatten_layer(relu2)

		#fully_connected1
		fc1, weightsf1, biasesf1 = self._fc_layer(
			flat, 
			num_features,
			self.FLAGS["fc_size"], 
		)

		#relu3
		relu3 = self._relu_layer(fc1)

		#fully_connected2
		fc2, weightsf2, biasesf2 = self._fc_layer(
			relu3,
			self.FLAGS["fc_size"],
			self.FLAGS["num_classes"]
		)

		#softmax
		y_pred = tf.nn.softmax(fc2)

		#prediction
		self.y_pred_cls = tf.argmax(y_pred, axis=1)

		#cost
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			logits=fc2, 
			labels=self.y_true
		)
		cost = tf.reduce_mean(cross_entropy)

		#optimization
		self.optimizer = tf.train.AdamOptimizer(
			learning_rate = self.FLAGS["learning_rate"]
		).minimize(cost)

		#performance measures
		self.correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)
		self.accuracy = tf.reduce_mean(
			tf.cast(self.correct_prediction, tf.float32)
		)

	
	def _fc_layer(self, input_layer, num_inputs, num_outputs):
		weights = self._weights(shape=[num_inputs, num_outputs])
		biases = self._biases(length=num_outputs)

		new_layer = input_layer @ weights + biases

		return new_layer, weights, biases

	def _flatten_layer(self, layer):
		layer_shape = layer.get_shape()
		num_features = layer_shape[1:4].num_elements()

		#-1 means total layer size will not change
		flat_layer = tf.reshape(layer, [-1, num_features])

		return flat_layer, num_features

	def _convolutional_layer(self, input_layer, num_inp_channels, \
								filter_size, num_filters):

		layer_shape = [filter_size, filter_size, num_inp_channels, num_filters]

		weights = self._weights(layer_shape)

		biases = self._biases(num_filters)

		new_layer = tf.nn.conv2d(
			input=input_layer, 
			filter=weights, 
			strides=[1, 1, 1, 1], 
			padding='SAME'
		)

		new_layer += biases

		return new_layer, weights

	def _relu_layer(self, input_layer):

		return tf.nn.relu(input_layer)

	def _max_pooling_layer(self, input_layer):
		return tf.nn.max_pool(
			value=input_layer, 
			ksize=[1, 2, 2, 1], 
			strides=[1, 2, 2, 1], 
			padding='SAME'
		)

	def _weights(self, shape):
		return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

	def _biases(self, length):
		return tf.Variable(tf.constant(0.05, shape=[length]))