import tensorflow as tf
from tensorflow import keras
import numpy as np
import GANVAE_Model
from GANVAE_Model import GANVAE, Monitor

"""
Script to run the GANVAE.
You can use the trained models or your own custom models or just use the one I made. The choice is yours.
Provide your models to GANVAE as parameters named- encoder, generator, and discriminator.
P.S. If you're using trained models make sure that z_dim = 128, because that's what I used while training.
"""

TRAIN_MODEL    = False
EPOCHS         = 50
BATCH_SIZE     = 32
z_dim          = 128
g_trainMul     = 1
init_lr_rate   = 3*10**-4
lr_decay_steps = 10**3
lr_decay_rate  = 0.98
beta_1         = 0.9

save_imgs   = True # Generate and save images after each epoch.
num_img_gen = 5
save_model  = False # save model after each epoch
invert_col  = False 

# MNIST Dataset
( x_train, y_train ), ( x_test, y_test ) = keras.datasets.mnist.load_data()
x_train = np.concatenate( [ x_train, x_test ], axis=0 )
x_train = np.expand_dims( x_train[:5000], -1 ) / 255.0

input_data  = tf.data.Dataset.from_tensor_slices( x_train ).batch( BATCH_SIZE )
input_shape = np.shape( x_train[0] )

# Trained Models
encoder, generator, discriminator = GANVAE_Model.load_trained_models()

if TRAIN_MODEL:
	ganvae = GANVAE(
		encoder=encoder, generator=generator, discriminator=discriminator,
		input_shape   = input_shape,
		z_dim         = z_dim,
		g_trainMul    = g_trainMul,
		d_thresh_low  = 0.55,
		d_thresh_high = 0.75
	)

	callbacks= [ Monitor( 
		save_imgs  = save_imgs,
		save_model = save_model,
		img_size   = input_shape,
		num_img    = num_img_gen,
		invert_col = invert_col,
	) ]

	# Learning rate scheduler to decay learning after certain number of steps
	lr_schedule = keras.optimizers.schedules.ExponentialDecay(
	   init_lr_rate, decay_steps=10**3, decay_rate=0.98 )
	ganvae.compile(
	   optimizer = keras.optimizers.Adam( learning_rate=lr_schedule, beta_1=beta_1 ),
	   loss_fn   = keras.losses.BinaryCrossentropy(),
	)

	history = ganvae.fit( input_data, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks )


GANVAE_Model.generate_images(
	generator,
	z_dim = z_dim, z_mean = 0, z_stddev = 1.0,
	num_img_sqr = 10,
	figsize     = 10,
	invert_col  = invert_col,
	img_size    = input_shape,
	save        = False,
	plot        = True
)
