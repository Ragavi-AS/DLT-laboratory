import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

BATCH_SIZE = 128
LATENT_DIM = 100

train_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*128, input_shape=(LATENT_DIM,)),
    tf.keras.layers.Reshape((7, 7, 128)),
    tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=2, padding='same', activation='tanh')
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        gen_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(gen_images, training=True)
        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        d_loss = loss_fn(tf.ones_like(real_output), real_output) + loss_fn(tf.zeros_like(fake_output), fake_output)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

for epoch in range(10):
    for images in train_data:
        train_step(images)
    print(f"Epoch {epoch+1} done")

noise = tf.random.normal([16, LATENT_DIM])
generated = generator(noise, training=False)

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.show()
