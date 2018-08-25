# seq2seq

Baseline encoder-decoder models in Tensorflow and [Dynet](https://dynet.readthedocs.io/en/latest/). Tensorflow implentation is based on @cgpotts Tensorflow RNN classifier for Stanford CS224U.
The original CS224U repository, which contains the base model and RNN classifier, is located [here](https://github.com/cgpotts/cs224u). Dynet implementation written in collaboration with @johntzwei, and contains a hierarchical-encoder model (HRED).

Code was written and tested for Python 3.6.5 with dependencies:

* Numpy 1.14.1
* Pandas 0.22.0
* Tensorflow 1.9.0
* Dynet 2.0

## Code hierarchy

`TfEncoderDecoder` is a subclass of `TfRNNClassifier`, which is a subclass of `TfModelBase`.
