# GAN+VAE with Tensorflow and Keras
This is my foray into the world of Generative Models.
I present to you a very basic GAN+VAE model inspired by Hardmaru's incredible blog,
["Generating Large Images from Latent Vectors"](https://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/) .  

• GANVAE_Model.py includes all the models and training algorithm to integrate and optimize all the models while training.  
• Run.py script is where you can train the model and generate images.  

I've tested the model in python 3.1 and tensorflow 2.1. Also make sure numpy and matplotlib are installed in your computer.

![Handwritten_Digit_Generation](https://user-images.githubusercontent.com/74816223/201645277-99f4ff5e-6143-47cd-865c-bf22b384ab45.gif)
![13x13_GeneratedDigits](https://user-images.githubusercontent.com/74816223/201643032-97686499-1205-42ef-8a6b-7c54a00083d6.png)

## Generative Adversarial Network - GAN
[Generative adversarial network](https://www.wikiwand.com/en/Generative_adversarial_network) or GAN for short is one of many generative models out there. This is where I personally started with image generation because The architecture is intuitive enough to understand and get a sense of what's going on behind the scenes, and complex enough to do some cool stuff.   
In a nutshell, There are 2 models, The generator and the discriminator. The generator generates an image, the discriminator then takes that generated image and an image from the dataset, and learns to discriminate between them. The job of the generator is to fool the discriminator into thinking that the generated image is from the dataset, and the discriminator's job is to successfully distinguish between them.  
The intricately choreographed dance between these two models is what helps them learn and become better at their job.  
Isn't it beautiful?  
I think it is.  

## Variational Autoencoder - VAE
[Variational Autoencoder](https://www.wikiwand.com/en/Variational_autoencoder) or VAE is another generative model used to, well, generate images. It takes the images from the dataset and compress them down to a vector, the so call Latent Vector or Latent Space. We've combined VAE with GAN to generate handwritten digits.
