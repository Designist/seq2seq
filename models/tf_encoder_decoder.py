import numpy as np
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
import warnings

__author__ = 'Nicholas Tomlin'

class TfEncoderDecoder(TfRNNClassifier):

	def __init__(self, **kwargs):
		super(TfEncoderDecoder, self).__init__(kwargs)

	def build_graph(self):
		self._define_embedding()

		encoder_lengths = tf.placeholder(tf.int32, [None])
		decoder_lengths = tf.placeholder(tf.int32, [None])

		encoder_outputs, encoder_state = self.encoding_layer(encoder_lengths)
		decoder_outputs, decoder_logits = self.decoding_layer(decoder_lengths, encoder_state)

		projection_layer = layers_core.Dense(self.vocab_size, use_bias=False)


	def encoding_layer(self, encoder_lengths):
		# Embedding:
		embedding_encoder = variable_scope.get_variable("embedding_encoder", [self.vocab_size, self.embed_dim])
		embedded_encoder_inputs = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs)

		# Build encoder RNN cell:
		encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Run the RNN:
		encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
			encoder_cell,
			embedded_encoder_inputs,
			sequence_length=encoder_lengths,
			time_major=True)

		return encoder_outputs, encoder_state


	def decoding_layer(self, decoder_lengths, encoder_state):
		# Embedding:
		embedding_decoder = variable_scope.get_variable("embedding_decoder", [self.vocab_size, self.embed_dim])
		embedded_decoder_inputs = tf.nn.embedding_lookup(embedding_decoder, self.decoder_inputs)

		# Build decoder RNN cell:
		decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Helper:
		helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding_input, decoder_lengths, time_major=True)
		
		# Decoder:
		decoder = tf.contrib.seq2seq.BasicDecoder(
		    decoder_cell,
		    helper,
		    encoder_state,
		    output_layer=projection_layer)

		# Dynamic decoding:
		decoder_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
		decoder_logits = decoder_outputs.rnn_output
		return decoder_outputs, decoder_logits

