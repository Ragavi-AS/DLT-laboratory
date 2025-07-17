import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data
english_sentences = ['hello', 'how are you', 'good morning', 'thank you']
french_sentences = ['bonjour', 'comment ça va', 'bonjour', 'merci']
french_sentences_input = ['<start> ' + sent for sent in french_sentences]
french_sentences_output = [sent + ' <end>' for sent in french_sentences]

# Tokenize English
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)
eng_seqs = eng_tokenizer.texts_to_sequences(english_sentences)
max_eng_len = max(len(seq) for seq in eng_seqs)
encoder_input_data = pad_sequences(eng_seqs, maxlen=max_eng_len, padding='post')

# Tokenize French
fra_tokenizer = Tokenizer(filters='')
fra_tokenizer.fit_on_texts(french_sentences_input + french_sentences_output)
fra_input_seqs = fra_tokenizer.texts_to_sequences(french_sentences_input)
fra_output_seqs = fra_tokenizer.texts_to_sequences(french_sentences_output)
max_fra_len = max(len(seq) for seq in fra_input_seqs)
decoder_input_data = pad_sequences(fra_input_seqs, maxlen=max_fra_len, padding='post')
decoder_output_data = pad_sequences(fra_output_seqs, maxlen=max_fra_len, padding='post')

# One-hot encode decoder output
num_fra_tokens = len(fra_tokenizer.word_index) + 1
decoder_output_onehot = tf.keras.utils.to_categorical(decoder_output_data, num_fra_tokens)

# Model
latent_dim = 256
encoder_inputs = Input(shape=(None,))
enc_embed_layer = Embedding(len(eng_tokenizer.word_index)+1, latent_dim)
encoder_embed = enc_embed_layer(encoder_inputs)
_, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embed)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
dec_embed_layer = Embedding(len(fra_tokenizer.word_index)+1, latent_dim)
decoder_embed = dec_embed_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
decoder_dense = Dense(num_fra_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_output_onehot, batch_size=2, epochs=500, verbose=0)

print("✅ Model trained.")

# Inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inf_inputs = Input(shape=(None,))
decoder_inf_embed = dec_embed_layer(decoder_inf_inputs)
decoder_inf_lstm, state_h_inf, state_c_inf = decoder_lstm(
    decoder_inf_embed, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_inf_outputs = decoder_dense(decoder_inf_lstm)

decoder_model = Model(
    [decoder_inf_inputs] + decoder_states_inputs,
    [decoder_inf_outputs] + decoder_states_inf)

# Reverse word index
reverse_fra_word_index = {i: w for w, i in fra_tokenizer.word_index.items()}
reverse_fra_word_index[0] = ''

def translate(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    states_value = encoder_model.predict(seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = fra_tokenizer.word_index['<start>']

    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_fra_word_index.get(sampled_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_fra_len:
            break
        decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Test translation
test_input = "hello"
print(f"English: {test_input}")
print(f"French : {translate(test_input)}")
