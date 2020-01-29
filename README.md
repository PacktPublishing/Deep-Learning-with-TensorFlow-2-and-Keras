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

## Incoming fixes:
This is a compiled list of errors reported to us through Amazon with their solutions and ETA for fixes. Thanks to Sam S. for highlighting these:

- [x] Chapter 5 Python source code files are missing on Github download webpage (**now fixed**)
- [x] Image files are missing (**added these files, also same issue as point 3. ETA 27/01/2020**)
- [x] On page 131 program, getting `cifar10_architecture.json` can't be opened error (**now fixed**)
- [x] On page 131 getting error with `.astype` (**now fixed**)
- [x] Many programs from Chapter 4 are giving errors and not running. Authors need to install latest versions of all the software tools on a clean new computer and test all the programs and update github web page with the source files that can be run using latest software tools versions please. Thanks.(**All the codes compile with the clean environment TF2.1. There are a few FutureWarnings which will be fixed. ETA 27/01/2020**)
- [ ] Python program on page 131 is not working (**An issue is raised which seems to be related to https://github.com/tensorflow/tensorflow/issues/35934.**)
- [ ] Please rename each source code file by prefixing with pgXXX_ corresponding to
approximate page number of the code. (**This is something we'll do once all the others issues are addressed.**)

*Thanks again Sam S. we really appreciate your feedback and we'd love to get in touch to say thank you personally! Please feel free if you would like to contact me as the book's Producer, Ben Renow-Clarke, at benc@packt.com be great to connect!*

![Antonio Gulli](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/images/Antonio.jpg) | ![Amita Kapoor](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/images/Amita.jpg) | ![Sujit Pal](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/images/Sujit.JPEG)
------ | ------ | ------
Antonio Gulli | Amita Kapoor | Sujit Pal


## Related Products
* [Python Machine Learning – Third Edition](https://www.packtpub.com/data/python-machine-learning-third-edition)

* [AI Crash Course](https://www.packtpub.com/data/ai-crash-course)

* [Dancing with Qubits](https://www.packtpub.com/data/dancing-with-qubits)
