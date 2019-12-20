import tensorflow as tf
import tensorflow_datasets as tfds

# See all registered datasets
builders = tfds.list_builders()
print (builders)

# Load a given dataset by name, along with the DatasetInfo
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']

print(info)