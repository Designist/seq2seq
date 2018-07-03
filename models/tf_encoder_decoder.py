import numpy as np
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
import warnings

__author__ = 'Nicholas Tomlin'

class TfEncoderDecoder(TfRNNClassifier):

	def __init__(self,
		max_input_length=5,
		max_output_length=5,
		**kwargs):
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
		super(TfEncoderDecoder, self).__init__(kwargs)

	def build_graph(self):
		self._define_embedding()

		
		encoder_lengths = tf.placeholder(tf.int32, [None])
		decoder_lengths = tf.placeholder(tf.int32, [None])

		self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, self.max_input_length])
		self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, self.max_output_length])

		self.projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

		encoder_outputs, encoder_state = self.encoding_layer(encoder_lengths)
		decoder_outputs, decoder_logits = self.decoding_layer(decoder_lengths, encoder_state)

		self.outputs = decoder_outputs
		self.model = decoder_logits


	def encoding_layer(self, encoder_lengths):
		# Embedding:
		embedding_encoder = tf.get_variable("embedding_encoder", [self.vocab_size, self.embed_dim])
		embedded_encoder_inputs = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs)

		# Build encoder RNN cell:
		encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Run the RNN:
		encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
			encoder_cell,
			embedded_encoder_inputs,
			sequence_length=encoder_lengths,
			time_major=True,
			dtype=tf.float32)

		return encoder_outputs, encoder_state


	def decoding_layer(self, decoder_lengths, encoder_state):
		# Embedding:
		embedding_decoder = tf.get_variable("embedding_decoder", [self.vocab_size, self.embed_dim])
		embedded_decoder_inputs = tf.nn.embedding_lookup(embedding_decoder, self.decoder_inputs)

		# Build decoder RNN cell:
		decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Helper:
		helper = tf.contrib.seq2seq.TrainingHelper(embedded_decoder_inputs, decoder_lengths, time_major=True)
		
		# Decoder:
		decoder = tf.contrib.seq2seq.BasicDecoder(
		    decoder_cell,
		    helper,
		    encoder_state,
		    output_layer=self.projection_layer)

		# Dynamic decoding:
		decoder_outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
		decoder_logits = decoder_outputs.rnn_output
		sample_id = decoder_outputs.sample_id
		return decoder_outputs, decoder_logits


	def prepare_output_data(self, y):
		return y


def simple_example():
    vocab = ['a', 'b', '$UNK']

    train = [
        [list('ab'), list('cde')],
        [list('aab'), list('ccde')],
        [list('abb'), list('cdde')],
        [list('aabb'), list('ccdde')],
        [list('ba'), list('dce')],
        [list('baa'), list('dcce')],
        [list('bba'), list('ddce')],
        [list('bbaa'), list('ddcce')]]

    test = [
        [list('aaab'), list('cccde')],
        [list('baaa'), list('dccce')]]

    mod = TfEncoderDecoder(
        vocab=vocab, max_iter=100, max_length=4)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()

