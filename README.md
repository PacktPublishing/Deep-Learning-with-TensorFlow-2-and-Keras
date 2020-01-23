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

1. Chapter 5 python source code files are missing on Github download webpage (**now fixed**)
2. Many programs in Chapter 4 are not working and giving many run-time errors (**working fine for us on TensorFlow 2, but we're still running these past the authors once more. ETA 27/01/2020**)
3. Python program on page 131 is not working (**Keras module imported through a deprecated module. Will push the updated code ETA 27/01/2020**)
4. Image files are missing (**added these files, also same issue as point 3. ETA 27/01/2020**)
5. On page 131 program, getting `cifar10_architecture.json` can't be opened error (**now fixed**)
6. On page 131 getting error with `.astype` (**same issue as point 3. ETA 27/01/2020**)
7. Please rename each source code file by prefixing with pgXXX_ corresponding to
approximate page number of the code. (**This is something we'll do once all the others issues are addressed. ETA - 31.01.20.**)
8. Many programs from Chapter 4 are giving errors and not running. (**working fine for us on TensorFlow 2, but we're still running these past the authors once more. ETA 27/01/2020**)
9. Therefore, authors need to install latest versions of all the software tools on a clean new computer and test all the programs and update github web page with the source files that can be run using latest software tools versions please. Thanks. (**working fine for us on TensorFlow 2, but we're still running these past the authors once more. ETA 27/01/2020**)

*Thanks again Sam S. we really appreciate your feedback and we'd love to get in touch to say thank you personally! Please feel free if you would like to contact me as the book's Producer, Ben Renow-Clarke, at benc@packt.com be great to connect!*

## Related Products
* [Python Machine Learning – Third Edition](https://www.packtpub.com/data/python-machine-learning-third-edition)

* [AI Crash Course](https://www.packtpub.com/data/ai-crash-course)

* [Dancing with Qubits](https://www.packtpub.com/data/dancing-with-qubits)
