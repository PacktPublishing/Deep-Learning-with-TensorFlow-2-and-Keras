# Deep Learning with TensorFlow 2 and Keras - 2nd Edition
This is the code repository for [Deep Learning with TensorFlow 2 and Keras - 2nd Edition](https://www.packtpub.com/data/deep-learning-with-tensorflow-2-0-and-keras-second-edition), published by [Packt](https://www.packtpub.com/). It contains all the supporting project files necessary to work through the book from start to finish.

## About the Book
Deep Learning with TensorFlow 2 and Keras, 2nd edition teaches deep learning techniques alongside TensorFlow (TF) and Keras. The book introduces neural networks with TensorFlow, runs through the main applications, covers two working example apps, and then dives into TF and cloudin production, TF mobile, and using TensorFlow with AutoML.

## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter 2.



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

## Install the latest version of conda, tensorflow, h5py, opencv
```
conda update conda
conda update --all
pip install --upgrade tensorflow
pip install --upgrade h5py
pip install opencv-python
```

## Incoming fixes:
This is a compiled list of errors reported to us through Amazon with their solutions and ETA for fixes. Thanks to Sam S. for highlighting these:

From Jan 26, 2020 Review Notes:
* Some programs are giving "cublas64-100.dll" file not found error. Is it possible for authors to zip this dll file and post it on this book's Github page please?
	
*This doesnt look like it's depend on autors, but on the specific setup used by the reader. An additional information is needed from reader on this.*

From Jan 25, 2020 Review Notes:
* Ran cifar10_predict.py program of page 131. It ran without errors and gave output results. However, the output gave [4 4]. This output is saying both "standing cat imge" and "dog image" belong to same class of four. This result may be wrong, due to one or both of the following reasons:
    - Model file "cifar10_weights.h5" used by this program is wrong?
    - Accuracy of training program that generated this model file is very low?

*All the above is fixed.*

Questions are:
* Which is the traing program that generated the above model file?
* Is it the program on pages 128 and 129?
 
*This is based on the pre-trained model contained in TF/Keras*

* Program on page 129 is saving to "model.h5" file.
* I ran the program on page 129 and renamed the model file "model.h5" as "cifar10_weights.h5".
* Then I ran program on page 131 and getting following error: ValueError: You are trying to load a weight file containing 13 layers into a model with 6 layers.
* Authors need to fix these errors please?
* Fix model file names of programs on pages 129 and 131 please?

*The READ.ME file in Chapter 4 clarifies how to install the model with pre-trained weights.*

*Thanks again Sam S. we really appreciate your feedback and we'd love to get in touch to say thank you personally! Please feel free if you would like to contact me as the book's Producer, Ben Renow-Clarke, at benc@packt.com be great to connect!*

![Antonio Gulli](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/images/Antonio.jpg) | ![Amita Kapoor](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/images/Amita.jpg) | ![Sujit Pal](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/images/Sujit.JPEG)
------ | ------ | ------
Antonio Gulli | Amita Kapoor | Sujit Pal


## Related Products
* [Python Machine Learning – Third Edition](https://www.packtpub.com/data/python-machine-learning-third-edition)

* [AI Crash Course](https://www.packtpub.com/data/ai-crash-course)

* [Dancing with Qubits](https://www.packtpub.com/data/dancing-with-qubits)
