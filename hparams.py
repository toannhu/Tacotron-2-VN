import tensorflow as tf
import numpy as np


# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners".
	cleaners='basic_cleaners',


	#Audio
	num_mels = 80,
	num_freq = 513, #only used when adding linear spectrograms post processing network
	rescale = True,
	rescaling_max = 0.999,
	trim_silence = True,
	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	use_lws=False,

	#Mel spectrogram
	fft_size = 1024,
	hop_size = 256,
	sample_rate = 22050, #22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms = None,

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True,
	allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max]

	#Limits
	min_level_db =- 100,
	ref_level_db = 20,
	fmin = 125,
	fmax = 7600,

	#Griffin Lim
	power = 1.55,
	griffin_lim_iters = 60,


	#Tacotron
	outputs_per_step = 2, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
	stop_at_any = True, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them

	embedding_dim = 512, #dimension of embedding space

	enc_conv_num_layers = 3, #number of encoder convolutional layers
	enc_conv_kernel_size = (5, ), #size of encoder convolution filters for each layer
	enc_conv_channels = 512, #number of encoder convolutions filters for each layer
	encoder_lstm_units = 256, #number of lstm units for each direction (forward and backward)

	smoothing = False, #Whether to smooth the attention normalization function
	attention_dim = 128, #dimension of attention space
	attention_filters = 32, #number of attention convolution filters
	attention_kernel = (31, ), #kernel size of attention convolution
	cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

	prenet_layers = [256, 256], #number of layers and number of units of prenet
	decoder_layers = 2, #number of decoder lstm layers
	decoder_lstm_units = 1024, #number of decoder lstm units on each layer
	max_iters = 2500, #Max decoder steps during inference (Just for safety from infinite loop cases)

	postnet_num_layers = 5, #number of postnet convolutional layers
	postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
	postnet_channels = 512, #number of postnet convolution filters for each layer

	mask_encoder = False, #whether to mask encoder padding while computing attention
	impute_finished = False, #Whether to use loss mask for padded sequences
	mask_finished = False, #Whether to mask alignments beyond the <stop_token> (False for debug, True for style)

	predict_linear = False, #Whether to add a post-processing network to the Tacotron to predict linear spectrograms (True mode Not tested!!)


	#Wavenet
	# Input type:
	# 1. raw [-1, 1]
	# 2. mulaw [-1, 1]
	# 3. mulaw-quantize [0, mu]
	# If input_type is raw or mulaw, network assumes scalar input and
	# discretized mixture of logistic distributions output, otherwise one-hot
	# input and softmax output are assumed.
	# **NOTE**: if you change the one of the two parameters below, you need to
	# re-run preprocessing before training.
	# **NOTE**: scaler input (raw or mulaw) is experimental. Use it your own risk.
	input_type="mulaw-quantize",
	quantize_channels=256,  # 65536 or 256

	silence_threshold=2,

	# Mixture of logistic distributions:
	log_scale_min=float(np.log(1e-14)),

	#TODO model params


	#Tacotron Training
	tacotron_batch_size = 16, #number of training samples on each training steps
	tacotron_reg_weight = 1e-6, #regularization weight (for l2 regularization)
	tacotron_scale_regularization = True, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

	tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	tacotron_start_decay = 50000, #Step at which learning decay starts
	tacotron_decay_steps = 50000, #starting point for learning rate decay (and determines the decay slope) (UNDER TEST)
	tacotron_decay_rate = 0.4, #learning rate decay rate (UNDER TEST)
	tacotron_initial_learning_rate = 1e-3, #starting learning rate
	tacotron_final_learning_rate = 1e-5, #minimal learning rate

	tacotron_adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
	tacotron_adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
	tacotron_adam_epsilon = 1e-6, #AdamOptimizer beta3 parameter

	tacotron_zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
	tacotron_dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet

	tacotron_teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs


	#Wavenet Training TODO



	#Eval sentences
	sentences = [
	'xin chào.'
	# 'xin chào.', 'tạm biệt.', 'hẹn gặp lại.', 'tôi nghĩ đề tài này thật tuyệt.', 'cảm ơn bạn đã góp ý cho chúng tôi.', 'mưa to tới rồi sấm chớp đùng đùng sợ quá.','quang mắt hí không phải vì mắt quang hí. mắt quang hí vì mỡ che bớt mắt quang.'
	]

	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
