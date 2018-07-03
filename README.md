# seq2seq

A simple encoder-decoder model based on @cgpotts Tensorflow RNN classifier for Stanford CS224U.
The original CS224U repository, which contains the base model and RNN classifier, is located [here](https://github.com/cgpotts/cs224u).
While writing this code I referenced the [official Tensorflow seq2seq tutorial](https://www.tensorflow.org/tutorials/seq2seq).

Code was written and tested for Python 3.6.5 with dependencies:

* Numpy 1.14.1
* Pandas 0.22.0
* Tensorflow 1.8.0

## Code hierarchy

`TfEncoderDecoder` is a subclass of `TfRNNClassifier`, which is a subclass of `TfModelBase`.
