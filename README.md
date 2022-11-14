# GAN+VAE with Tensorflow and Keras
This is my foray into the world of Generative Models.
I present to you a very basic GAN+VAE model inspired by Hardmaru's incredible blog,
["Generating Large Images from Latent Vectors"](https://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/) .  

• GANVAE_Model.py includes all the models and training algorithm to integrate and optimize all the models while training.  
• Run.py script is where you can train the model and generate images.  

I've tested the model in python 3.1 and tensorflow 2.1. Also make sure numpy and matplotlib is installed in your computer.

![Handwritten Digit Generation](https://user-images.githubusercontent.com/74816223/201641046-2b99337a-433c-4427-a81f-da89bc7fe257.gif)

![15x15_GeneratedDigits](https://user-images.githubusercontent.com/74816223/201462211-fce26c0d-3d57-4fe4-ade5-fd15d64ef395.png)

## Generative Adversarial Network - GAN
[Generative adversarial network](https://www.wikiwand.com/en/Generative_adversarial_network) or GAN is one of many generative models out there. This is where I personally started with image generation because The architecture is intuitive enough to understand and get a sense of what's going on behind the scenes, and complex enough to do some cool stuff.

## Variational Autoencoder - VAE
[Variational Autoencoder](https://www.wikiwand.com/en/Variational_autoencoder) or VAE is another generative model used to, well, generate images. It takes the images from the dataset and compress them down to a vector, so call Latent Vector or Latent Space. We've combine VAE with GAN to generate hand-written digits.
