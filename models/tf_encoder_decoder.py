import numpy as np
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
import warnings

__author__ = 'Nicholas Tomlin'

class TfEncoderDecoder(TfRNNClassifier):
	'''
	Input and output vocabulary should be the same for response model.
	'''

	def __init__(self,
		max_input_length=5,
		max_output_length=5,
		**kwargs):
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
		super(TfEncoderDecoder, self).__init__(**kwargs)


	def build_graph(self):
		self._define_embedding()
		self._init_placeholders()
		self._init_embedding()

		self.encoding_layer()
		self.decoding_layer()


	def _init_placeholders(self):
		"""
		Helper function for build_graph which initializes seq2seq
		placeholders for encoder inputs and decoder targets
		"""
		self.encoder_inputs = tf.placeholder(
			shape=[None, self.max_input_length],
			dtype=tf.int32,
			name="encoder_inputs")

		self.encoder_lengths = tf.placeholder(
			shape=[None],
			dtype=tf.int32,
			name="encoder_lengths")

		self.decoder_targets = tf.placeholder(
			shape=[None, self.max_output_length],
			dtype=tf.int32,
			name="decoder_targets")

		self.decoder_lengths = tf.placeholder(
			shape=[None],
			dtype=tf.int32,
			name="decoder_lengths")

	def _init_embedding(self):
		embedding_encoder = tf.Variable(tf.random_uniform(
			shape=[self.vocab_size, self.embed_dim],
			minval=-1.0,
			maxval=1.0,
			name="embedding_encoder"))
		embedded_encoder_inputs = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs)

		embedding_decoder = tf.Variable(tf.random_uniform(
			shape=[self.vocab_size, self.embed_dim],
			minval=-1.0,
			maxval=1.0,
			name="embedding_decoder"))
		embedded_decoder_targets = tf.nn.embedding_lookup(embedding_decoder, self.decoder_targets)

	def encoding_layer(self):
		# Build encoder RNN cell:
		encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Run the RNN:
		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
			cell=encoder_cell,
			inputs=embedded_encoder_inputs,
			sequence_length=self.encoder_lengths,
			time_major=True,
			dtype=tf.float32)

		self.encoder_final_state = encoder_final_state

	def decoding_layer(self):
		# Build decoder RNN cell:
		decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, activation=self.hidden_activation)

		# Helper:
		helper = tf.contrib.seq2seq.TrainingHelper(embedded_decoder_targets, self.decoder_lengths, time_major=True)

		# Projection layer:
		projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)
		
		# Decoder:
		decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=decoder_cell,
			helper=helper,
			initial_state=self.encoder_final_state,
			output_layer=projection_layer)

		# Dynamic decoding:
		decoder_outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
		decoder_logits = decoder_outputs.rnn_output
		sample_id = decoder_outputs.sample_id
		
		self.outputs = decoder_outputs
		self.model = decoder_logits


	def fit(self, X, y, **kwargs):
		"""
		Modified fit() for sequence modelling.
		"""
		if isinstance(X, pd.DataFrame):
			X = X.values

		# Incremental performance:
		X_dev = kwargs.get('X_dev')
		if X_dev is not None:
			dev_iter = kwargs.get('test_iter', 10)

		# Start the session:
		tf.reset_default_graph()
		self.sess = tf.InteractiveSession()

		# Build the computation graph. This method is instantiated by
		# individual subclasses. It defines the model.
		self.build_graph()

		# Optimizer set-up:
		self.cost = self.get_cost_function()
		self.optimizer = self.get_optimizer()

		# Initialize the session variables:
		self.sess.run(tf.global_variables_initializer())

		# Training, full dataset for each iteration:
		for i in range(1, self.max_iter+1):
			loss = 0
			for X_batch, y_batch in self.batch_iterator(X, y):
				_, batch_loss = self.sess.run(
					[self.optimizer, self.cost],
					feed_dict=self.train_dict(X_batch, y_batch))
				loss += batch_loss
			self.errors.append(loss)
			if X_dev is not None and i > 0 and i % dev_iter == 0:
				self.dev_predictions.append(self.predict(X_dev))
			if loss < self.tol:
				self._progressbar("stopping with loss < self.tol", i)
				break
			else:
				self._progressbar("loss: {}".format(loss), i)
		return self


	def get_cost_function(self, **kwargs):
		"""Uses `softmax_cross_entropy_with_logits` so the
		input should *not* have a softmax activation
		applied to it.
		"""
		return tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(
				logits=self.model,
				labels=tf.one_hot(self.outputs, depth=vocab_size, dtype=tf.float32)))


	def predict(self, X):
		decoder_prediction = tf.argmax(self.model, 2)
		predictions = sess.run(
			decoder_prediction,
			feed_dict={
				encoder_inputs: X,
				decoder_inputs: din_,
			})

		return predictions


	def train_dict(self, X, y):
		X, _ = self._convert_X(X)
		y, _ = self._convert_X
		return {self.encoder_inputs: X, self.decoder_inputs: y}


	def test_dict(self, X):
		X, _ = self._convert_X(X)
		return {self.encoder_inputs: X}


def simple_example():
	vocab = ['a','b','c']
	seq2seq = TfEncoderDecoder(vocab=vocab)
	seq2seq.build_graph()

if __name__ == '__main__':
	simple_example()

