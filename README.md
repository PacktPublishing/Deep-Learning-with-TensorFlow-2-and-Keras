# Deep Learning with TensorFlow 2 and Keras - 2nd Edition
This is the code repository for [Deep Learning with TensorFlow 2 and Keras - 2nd Edition](https://www.packtpub.com/data/deep-learning-with-tensorflow-2-0-and-keras-second-edition), published by [Packt](https://www.packtpub.com/). It contains all the supporting project files necessary to work through the book from start to finish.

## About the Book
Deep Learning with TensorFlow 2 and Keras, 2nd edition teaches deep learning techniques alongside TensorFlow (TF) and Keras. The book introduces neural networks with TensorFlow, runs through the main applications, covers two working example apps, and then dives into TF and cloudin production, TF mobile, and using TensorFlow with AutoML.

## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
@tf.function
def fn(input, state):
    return cell(input, state)

input = tf.zeros([100, 100])
state = [tf.zeros([100, 100])] * 2
# warmup
cell(input, state)
fn(input, state)

```


## Related Products
* [Python Machine Learning – Third Edition](https://www.packtpub.com/data/python-machine-learning-third-edition)

* [AI Crash Course](https://www.packtpub.com/data/ai-crash-course)

* [Dancing with Qubits](https://www.packtpub.com/data/dancing-with-qubits)