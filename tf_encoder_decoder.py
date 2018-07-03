import numpy as np
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
import warnings

__author__ = 'Nicholas Tomlin'

class TfEncoderDecoder(TfRNNClassifier):

	def __init__(self, **kwargs):
        super(TfEncoderDecoder, self).__init__(**kwargs)

	def build_graph(self):
		self._define_embedding()

		# Embedding:
		embedding_encoder = variable_scope.get_variable("embedding_encoder", [self.vocab_size, self.embed_dim])
		embedding_decoder = variable_scope.get_variable("embedding_decoder", [self.vocab_size, self.embed_dim])

		encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs)
		decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, self.decoder_inputs)

		# Build encoder RNN cell:
		encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Run the RNN:

		# Build decoder RNN cell:
		decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)