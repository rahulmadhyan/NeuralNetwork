from datetime import datetime
import numpy as np 
import copy

class Dataset:
	@staticmethod
	#loads data from file
	def load_data_from_file(filename):
		date, latitude, longitude, magnitude = [], [], [], []

		with open(filename, "r") as f:
			f.readline() #skips the first line

			for line in f:
				elements = line.split(',')
				try:
					date.append(datetime.strptime(f"{elements[0]} {elements[1]}", "%m/%d/%Y %H:%M:%S"))
					latitude.append(float(elements[2]))
					longitude.append(float(elements[3]))
					magnitude.append(float(elements[8]))
				except ValueError:
					pass

		return np.array(date), np.array(latitude), np.array(longitude), np.float32(magnitude)

	@staticmethod
	def normalize_date(array):
		min_date = min(array)
		max_date = max(array)
		delta = max_date - min_date

		return np.float32([(d - min_date).total_seconds() / delta.total_seconds() for d in array])

	@staticmethod
	def normalize_coordinates(latitude, longitude):

		latitude_radians = np.deg2rad(latitude)
		longitude_radians = np.deg2rad(longitude)

		x = np.cos(latitude_radians) * np.cos(longitude_radians)
		y = np.cos(latitude_radians) * np.sin(longitude_radians)
		z = np.sin(latitude_radians)

		return x, y, z	

	#transforms the date, latitude and longtitude arrays in vectors to be fed into the NN	
	@staticmethod
	def vectorize(date, latitude, longitude):	

		return np.concatenate(Dataset.normalize_coordinates(latitude, longitude) + (Dataset.normalize_date(date),))\
            .reshape((4, len(date)))\
            .swapaxes(0, 1)

class Math:
    @staticmethod
    #sigmoid activation function and its derivative
    def sigmoid(x, deriv=False):
        
        if deriv:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    @staticmethod
    #rectifier linear activation function and its derivative
    def rect_linear(x, deriv=False):
    
        if deriv:
            return np.ones_like(x) * (x > 0)

        return x * (x > 0)

    #returns random parameter within distance    
    @staticmethod
    def random_parameter(x, min, max, distance):
        
        alpha = 2 * np.random.random() - 1
        new = x + distance * alpha

        if new < min:
            return min
        elif new > max:
            return max

        return new

class Generator:
	#returns radnom batches of given X & Y arrays of specified size
    @staticmethod
    def get_random_batch(size, X, Y):
        
        while True:
            index = np.arange(X.shape[0])
            np.random.shuffle(index)

            s_X, s_Y = X[index], Y[index]
            for i in range(X.shape[0] // size):
                yield (X[i * size:(i + 1) * size], Y[i * size:(i + 1) * size])

    #splits arrays into smaller batches            
    @staticmethod
    def get_batch(size, X, Y):
        
        if X.shape[0] % size != 0:
            print("[/!\ Warning /!\] the full set will not be executed because of a poor choice of batch_size")

        for i in range(X.shape[0] // batch_size):
            yield X[i * size:(i + 1) * size], Y[i * size:(i + 1) * size]          

#main 
if __name__ == "__main__":
	#load paramaters from file
	date, latitude, longitude, magnitude = Dataset.load_data_from_file("database.csv")
	data_size = len(date)

	#vectorize data
	vectorX, vectorY = Dataset.vectorize(date, latitude, longitude), magnitude.reshape((data_size, 1))

	#split vectors into train & cross-validation sets
	cv_set_size = int(0.1 * data_size)
	index = np.arange(data_size)
	np.random.shuffle(index)
	trainX, trainY = vectorX[index[cv_set_size:]], vectorY[index[cv_set_size:]]
	cvX, cvY = vectorX[index[:cv_set_size]], vectorY[index[cv_set_size:]]

	#random initialization of weights
	weight0_origin = 2 * np.random.random((trainX.shape[1], 32)) - 1
	weight1_origin = 2 * np.random.random((32, trainY.shape[1])) - 1

	#placeholder for hyperparameters
	best_error = 9999
	best_learning_rate = -3
	best_momentum = 0.9
	best_batch_size = 64
	best_max_epochs_log = 4 
	learning_rate_log = None
	momentum = None
	batch_size = None
	max_epochs_log = None

	for i in range(50):
		#hyperparameters
		learning_rate_log = Math.random_parameter(best_learning_rate, -5, -1, 0.5)
		momentum = Math.random_parameter(best_momentum, 0.5, 0.95, 0.1)
		batch_size = np.int64(Math.random_parameter(best_batch_size, 10, 128, 10))
		max_epochs_log = Math.random_parameter(best_max_epochs_log, 3, 5, 0.5)

		learning_rate = np.power(10, learning_rate_log)
		max_epochs = np.int64(np.power(10, max_epochs_log))
					
		#display hyperparameters
		print(f"Iteration: {i}")
		print(f"Learning Rate: {learning_rate}")
		print(f"Momentum: {momentum}")
		print(f"Batch Size: {batch_size}")
		print(f"Max Epochs: {max_epochs}")

        #Reset weights
		weight0 = copy.deepcopy(weight0_origin)
		weight1 = copy.deepcopy(weight1_origin)

        #initialize momentum 
		momentum_weight0 = np.zeros_like(weight0)
		momentum_weight1 = np.zeros_like(weight1)

		#get batch generator
		batch_generator = Generator.get_random_batch(batch_size, trainX, trainY)

		#train model 
		for j in range(max_epochs):
			batch = next(batch_generator)

			#feed forward propogation
			layer0 = batch[0]
			layer1 = Math.sigmoid(np.dot(layer0, weight0))
			layer2 = Math.sigmoid(np.dot(layer1, weight1))

			layer2_error = batch[1] - layer2
			layer2_delta = layer2_error * Math.rect_linear(layer2, deriv=True)

			layer1_error = layer2_delta.dot(weight1.T)
			layer1_delta = layer1 * Math.sigmoid(layer1, deriv=True)

			momentum_weight1 = momentum * momentum_weight1 + layer1.T.dot(layer2_delta) * learning_rate
			momentum_weight0 = momentum * momentum_weight0 + layer0.T.dot(layer1_delta) * learning_rate

			weight1 += momentum_weight1
			weight0 += momentum_weight0

			#evaluate model
		current_error = 0
		for batch in Generator.get_batch(10, cvX, cvY):
			layer0 = batch[0]
			layer1 = Math.sigmoid(np.dot(layer0, weight0))
			layer2 = Math.sigmoid(np.dot(layer1, weight1))

			current_error += np.sum(np.abs(batch[1] - layer2))

		current_error /= cv_set_size

		print(f"Error: {current_error}\n")

		if current_error < best_error:
		    best_error = current_error
		    best_learning_rate_log = learning_rate_log
		    best_momentum = momentum
		    best_batch_size = batch_size
		    best_max_epochs_log = max_epochs_log

		#display best hyper parameters for current iteration
		print(f"Best Error: {best_error}")
		print(f"Best Learning Rate: {best_learning_rate_log}")
		print(f"Best Momentum: {best_momentum}")
		print(f"Best Batch Size: {best_batch_size}")
		print(f"Best Max Epochs: {best_max_epochs_log}\n")