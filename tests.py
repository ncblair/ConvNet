



class TestNeuralNetMethods(unittest.TestCase):

	def testTrainEasy(self):
		image_length = 10
		num_channels = 1
		threshold = image_length **2 * num_channels // 2
		train_data = np.random.rand(
			10000, image_length**2, num_channels
		).astype("float32")

		test_data = np.random.rand(
			1000, image_length**2, num_channels
		).astype("float32")

		train_labels = np.array(
			[[int(np.sum(x) > threshold), int(np.sum(x) <= threshold)] \
			for x in train_data]
		)

		test_labels = np.array(
			[[int(np.sum(x) > threshold), int(np.sum(x) <= threshold)] \
			for x in test_data]
		)

		args = {
			"image_length" : image_length,
			"num_channels" : num_channels,
			"num_epochs" : 10
		}

		neural_net = Net(
			x=train_data, 
			y_true=train_labels, 
			load=False, 
			name="easy_test", 
			args=args
		)

		predictions = neural_net.classify(test_data)

		accuracy = 1 - (
			np.sum(
				np.abs(predictions - np.argmax(test_labels, axis=1))
			) / len(predictions)
		)

		print(f"accuracy: {accuracy}")
		neural_net.close()
		print(predictions)
		print(np.argmax(test_labels, axis=1))
		self.assertTrue(accuracy > .8)
		return accuracy

	def testTrainMnist(self):
		from tensorflow.examples.tutorials.mnist import input_data
		data = input_data.read_data_sets('data/MNIST/', one_hot=True)
		data.test.cls = np.argmax(data.test.labels, axis=1)
		train_data = data.train.images

		train_data = np.array([[[x] for x in row] for row in train_data])
		train_labels = data.train.labels

		test_data = data.test.images
		test_data = np.array([[[x] for x in row] for row in test_data])
		test_labels = data.test.labels
		test_class = data.test.cls

		num_tested = len(data.test.images)

		args = {
			"image_length" : 28, 
			"num_channels" : 1, 
			"num_classes" : 10
		}

		neural_net = Net(
			x=train_data, 
			y_true=train_labels, 
			load=False, 
			name="mnist_test", 
			args=args
		)

		predictions = neural_net.classify(test_data)
		num_correct = np.sum(predictions == test_class)
		accuracy = num_correct / len(data.test.images)

		print(f"accuracy: {accuracy}")
		neural_net.close()
		print(predictions)
		print(np.argmax(test_labels, axis=1))
		self.assertTrue(accuracy > .90)
		return accuracy

	def testLoad(self):
		#run an easy test
		acc1 = self.testTrainEasy()

		#load the easy test
		neural_net = Net(
			load=True,
			train=False, 
			name="easy_test"
		)

		predictions = neural_net.classify(test_data)
		num_correct = np.sum(predictions == test_class)
		accuracy = num_correct / len(data.test.images)

		print(f"accuracy: {accuracy}")
		neural_net.close()
		print(predictions)
		print(np.argmax(test_labels, axis=1))
		self.assertEqual(accuracy, acc1)
		return accuracy



if __name__ == '__main__':
	unittest.main()