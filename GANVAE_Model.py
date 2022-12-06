import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
curr_path = os.path.dirname( os.path.abspath( __file__ ) )

gen_imgs_file                 = curr_path+'/SampleImages/img_{}%03d.png'.format( time.strftime( '%H%M%S' ) )
discriminator_savedModel_file = curr_path+'/TrainedModels/Discriminator_Trained_Model'
generator_savedModel_file     = curr_path+'/TrainedModels/Generator_Trained_Model'
vae_savedModel_file           = curr_path+'/TrainedModels/VAE_Trained_Model'

class GANVAE( keras.Model ):
   def __init__(
      self,
      encoder       = None,
      generator     = None,
      discriminator = None,
      input_shape   = ( 28,28,1 ),
      z_dim         = 128,
      g_trainMul    =  1,
      d_thresh_low  = 0,
      d_thresh_high = 1
   ):
      super(GANVAE, self).__init__()
      """
      ENCODER: Encoder takes in an image from dataset as input, and tries to compress all of that
         dense information down into a vector — so called "Latent vector" or "Latent space" — the
         dimension on which is denoted by "z_dim".

      GENERATOR: Generator takes as input the latent vector generated by the encoder and tries to
         generates an image that resembles the original image that was the input of the encoder.

      DISCRIMINATOR: Discriminator is here to classify if an image is from the dataset or generated by the generator.

      INPUT_SHAPE: Shape/Dimension of the images in dataset.

      Z_DIM: Dimension of our latent vector.

      G_TRAINMUL: The number of times to train generator for each train step of discriminator.

      D_THRESH_LOW: Threshold to stop the training of discriminator if it's getting really good at
         discriminating between real and generated images, i.e it's loss is low.

      D_THRESH_HIGH: Again to stop the training of the discriminator if the generator is not doing
         a good job of fooling the discriminator, i.e generator's loss is high.
      """
      self.encoder       = Encoder( z_dim, input_shape )   if encoder      == None else encoder
      self.generator     = Generator( z_dim, input_shape ) if generator    == None else generator
      self.discriminator = Discriminator( input_shape )    if discriminator== None else discriminator
      self.z_dim         = z_dim
      self.g_trainMul    = g_trainMul
      self.d_thresh_low  = d_thresh_low
      self.d_thresh_high = d_thresh_high

   def compile( self, optimizer, loss_fn ):
      super(GANVAE, self).compile(run_eagerly=True)
      """
      OPTIMIZER: Think of the optimizer as just an algorithm that shows our models the way
         towards the best possible result we're looking for.
      LOSS_FN: Loss function tells the optimizer if it's going in the right direction.
      """
      self.optimizer = optimizer
      self.loss_fn   = loss_fn
      """
      These below are some metrics that'll be exposed in the terminal during the training 
      so we can assess how the model is behaving.
      """
      self.d_loss_metric   = keras.metrics.Mean(name='d_loss')
      self.g_loss_metric   = keras.metrics.Mean(name='g_loss')
      self.kl_loss_metric  = keras.metrics.Mean(name='kl_loss')
      self.vae_loss_metric = keras.metrics.Mean(name='vae_loss')

   def Summary(self):
      print( self.encoder.summary(), self.generator.summary(), self.discriminator.summary() )
   @property
   def metrics(self):
      return [ self.d_loss_metric, self.g_loss_metric, self.kl_loss_metric, self.vae_loss_metric ]

   def train_step(self, real_imgs):
      """
      This is where we train our models.
      # 1: Input images (real_imgs) from dataset into VAE (encoder).
         It'll return our latent vector ( z ), the mean and variance of it's probability distribution.
      # 2: Input z (latent vector output by encoder) into the generator. it'll learn to decode the z and generate an image.
      # 3: Calculate all the losses.
      # 4: Calculate relevant gradients and apply them to the weights and biases of our models.
      """
      batch_size = tf.shape(real_imgs)[0]

      # Train VAE and Generator
      for i in range(self.g_trainMul):
         with tf.GradientTape( persistent=True ) as tape:
         # 1:
            z_mean, z_log_var, z = self.encoder( real_imgs )
         # 2:
            gen_img = self.generator( z )
         # 3:
            reconstruction_loss = tf.reduce_mean(
               tf.reduce_sum( keras.losses.binary_crossentropy( real_imgs, gen_img ), axis=(1,2) )
            )

            g_loss   = self.loss_fn( tf.ones( ( batch_size, 1 ) ), self.discriminator( gen_img ) )
            kl_loss  = -0.5 * ( 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var) )
            kl_loss  = tf.reduce_mean( tf.reduce_sum( kl_loss, axis=1 ) )
            vae_loss = reconstruction_loss + kl_loss
      # 4:
         # dvae_loss/dW_e
         grads = tape.gradient( vae_loss, self.encoder.trainable_weights )
         self.optimizer.apply_gradients( zip( grads, self.encoder.trainable_weights ) )

         # dvae_loss/dW_g
         grads = tape.gradient( vae_loss, self.generator.trainable_weights )
         self.optimizer.apply_gradients( zip( grads, self.generator.trainable_weights ) )

         # dg_loss/dW_g
         grads = tape.gradient( g_loss, self.generator.trainable_weights )
         self.optimizer.apply_gradients( zip( grads, self.generator.trainable_weights ) )

         # dg_loss/dW_e
         grads = tape.gradient( g_loss, self.encoder.trainable_weights )
         self.optimizer.apply_gradients( zip( grads, self.encoder.trainable_weights ) )

      # Train discriminator
      with tf.GradientTape() as tape:
         z          = tf.random.normal( shape=( batch_size, self.z_dim ), mean=0.0, stddev=1.0 )
         gen_img    = self.generator( z )
         # Discrimination's prediction of real and generated images
         real_pred  = self.discriminator( real_imgs )
         gen_pred   = self.discriminator( gen_img )
         d_realLoss = self.loss_fn( tf.ones( batch_size, 1 ), real_pred )
         d_genLoss  = self.loss_fn( tf.zeros( batch_size, 1 ), gen_pred )
         # Average loss
         d_loss     = ( d_realLoss + d_genLoss ) * 0.5
      """
      Threshold discriminator so that it doesn't get too good at classifying real from generated images and 
      generator is compatible with it.
      """
      if ( g_loss.numpy() < self.d_thresh_high ) & ( d_loss.numpy() > self.d_thresh_low ):
         grads = tape.gradient( d_loss, self.discriminator.trainable_weights )
         self.optimizer.apply_gradients( zip( grads, self.discriminator.trainable_weights ) )

      # Update metrics
      self.d_loss_metric.update_state(d_loss)
      self.g_loss_metric.update_state(g_loss)
      self.kl_loss_metric.update_state(kl_loss)
      self.vae_loss_metric.update_state(vae_loss)

      return {
         'd'  : self.d_loss_metric.result(),
         'g'  : self.g_loss_metric.result(),
         'vae': self.vae_loss_metric.result(),
         'kl' : self.kl_loss_metric.result(),
      }

# Inside Monitor class, You can do all sort of stuff at the end of each epoch while training
class Monitor( keras.callbacks.Callback ):
   def __init__( self, num_img=3, z=None, img_size=(28,28,1), save_imgs=True, save_model=True, invert_col=False ):
      self.num_img    = num_img
      self.z          = z
      self.img_size   = img_size
      self.save_imgs  = save_imgs
      self.save_model = save_model
      self.invert_col = invert_col

   def on_epoch_end( self, epoch, logs=None ):
      if self.save_model:
         self.model.discriminator.save( discriminator_savedModel_file )
         self.model.generator.save( generator_savedModel_file )
         self.model.encoder.save( vae_savedModel_file )

      if self.save_imgs:
         generate_images(
            self.model.generator,
            z           = self.z,
            z_dim       = self.model.z_dim,
            num_img_sqr = self.num_img,
            epoch       = epoch,
            figsize     = 10,
            invert_col  = False,
            img_size    = self.img_size,
            save        = True,
            plot        = False,
         )

class Sampling( layers.Layer ):
   def call(self, inputs):
      z_mean, z_log_var = inputs
      batch   = tf.shape(z_mean)[0]
      dim     = tf.shape(z_mean)[1]
      epsilon = tf.random.normal( shape=(batch, dim), mean=0.0, stddev=1.0 )
      return z_mean + tf.exp( 0.5 * z_log_var ) * epsilon

def Encoder( z_dim=5, input_shape=( 28,28,1 ) ):
   inputs = keras.Input( shape=input_shape )

   x = layers.Conv2D( 32, 3, activation='relu', strides=2, padding='same' )(inputs)
   x = layers.Conv2D( 62, 3, activation='relu', strides=2, padding='same' )(x)
   x = layers.Conv2D( 128, 3, activation='relu', strides=2, padding='same' )(x)
   x = layers.Conv2D( 256, 3, activation='relu', strides=2, padding='same' )(x)
   x = layers.Flatten()(x)
   x = layers.Dense( 32, activation='relu' )(x)

   z_mean    = layers.Dense( z_dim, name='z_mean' )(x)
   z_log_var = layers.Dense( z_dim, name='z_log_var' )(x)
   z         = Sampling()( [ z_mean, z_log_var ] )

   encoder = keras.Model( inputs, [z_mean, z_log_var, z], name='Encoder' )
   return encoder

def Generator( z_dim, output_shape=(28,28,1) ):
    inputs = keras.Input( shape=(z_dim,) )

    x = layers.Dense( output_shape[0] * output_shape[0] * 32, activation='relu' )(inputs)
    x = layers.Reshape( ( output_shape[0], output_shape[0], 32 ) )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU( alpha=0.3 )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2DTranspose( 32, kernel_size=4, strides=1, padding='same', kernel_regularizer=keras.regularizers.L2(l2=0.001) )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU( alpha=0.2 )(x)
    x = layers.Dropout( 0.2 )(x)

    x = layers.Conv2DTranspose(64, kernel_size=4, strides=1, padding='same', kernel_regularizer=keras.regularizers.L2(l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2DTranspose(128, kernel_size=4, strides=1, padding='same', kernel_regularizer=keras.regularizers.L2(l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Conv2D( output_shape[2], kernel_size=4, padding='same', activation='sigmoid' )(x)
    generator = keras.Model( inputs, output, name = 'Generator')
    return generator

def Discriminator( input_shape ):
   inputs = keras.Input(shape=input_shape)

   x = layers.Conv2D( 32, kernel_size=4, strides=2, padding='same', kernel_regularizer=keras.regularizers.L2(0.01) )(inputs)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.MaxPooling2D()(x)
   x = layers.Dropout(0.2)(x)

   x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_regularizer=keras.regularizers.L2(0.01))(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   x = layers.MaxPooling2D()(x)
   x = layers.Dropout(0.2)(x)

   x = layers.Flatten()(x)
   x = layers.Dense(8, activation='relu')(x)

   output = layers.Dense(1, activation='sigmoid')(x)
   discriminator =  keras.Model(inputs, output, name='Discriminator')
   return discriminator

# Generate/plot images in a grid
def generate_images(
   generator,
   z           = None,
   z_dim       = 128,
   z_mean      = 0,
   z_stddev    = 1.0,
   num_img_sqr = 30,
   epoch       = 1,
   figsize     = 10,
   save        = False,
   plot        = True,
   invert_col  = False,
   img_size    = ( 28,28,1 ) 
):
   n = num_img_sqr
   scale  = 1.0
   figure = np.zeros( ( img_size[0] * n, img_size[0] * n, img_size[2] ) )
   grid_x = np.linspace( 1, n, n )
   grid_y = np.linspace( 1, n, n )[::-1]

   for i, yi in enumerate( grid_y ):
      for j, xi in enumerate( grid_x ):
         z_new     = tf.random.normal( shape=( 1,z_dim ), mean=z_mean, stddev=z_stddev ) if z == None else z
         x_decoded = generator.predict( z_new )
         digit     = x_decoded[0].reshape( img_size[0], img_size[0], img_size[2] )
         figure[
            i * img_size[0] : ( i + 1 ) * img_size[0],
            j * img_size[0] : ( j + 1 ) * img_size[0],
         ] = digit
   
   figure = ( 1-figure )*255.0 if invert_col else figure*255.0
   if save:
      img = keras.preprocessing.image.array_to_img( figure )
      img.save( gen_imgs_file % epoch )

   if plot:
      plt.figure( figsize =( figsize, figsize ) )

      start_range    = img_size[0] // 2
      end_range      = n * img_size[0] + start_range
      pixel_range    = np.arange( start_range, end_range, img_size[0] )
      sample_range_x = np.round( grid_x, 1 )
      sample_range_y = np.round( grid_y, 1 )

      plt.xticks( pixel_range, sample_range_x )
      plt.yticks( pixel_range, sample_range_y )
      plt.imshow( figure, cmap='Greys_r' )
      plt.show()

def load_trained_models():
   encoder       = keras.models.load_model( vae_savedModel_file )
   generator     = keras.models.load_model( generator_savedModel_file )
   discriminator = keras.models.load_model( discriminator_savedModel_file )
   return encoder, generator, discriminator