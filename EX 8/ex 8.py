import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

questions = ['hi', 'how are you', 'what is your name', 'hello', 'what do you do']
answers = ['<start> hello <end>', '<start> i am fine <end>', '<start> i am a chatbot <end>',
           '<start> hi <end>', '<start> i chat with you <end>']

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

q_seqs = tokenizer.texts_to_sequences(questions)
a_seqs_in = tokenizer.texts_to_sequences([a.rsplit(' ', 1)[0] for a in answers])
a_seqs_out = tokenizer.texts_to_sequences([a.split(' ', 1)[1] for a in answers])

max_q_len = max(len(seq) for seq in q_seqs)
max_a_len = max(len(seq) for seq in a_seqs_in)

q_seqs = pad_sequences(q_seqs, maxlen=max_q_len, padding='post')
a_seqs_in = pad_sequences(a_seqs_in, maxlen=max_a_len, padding='post')
a_seqs_out = pad_sequences(a_seqs_out, maxlen=max_a_len, padding='post')

a_out_oh = tf.keras.utils.to_categorical(a_seqs_out, VOCAB_SIZE)

EMBEDDING_DIM = 64
LSTM_UNITS = 128

encoder_inputs = Input(shape=(max_q_len,))
enc_emb = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
encoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

decoder_inputs = Input(shape=(max_a_len,))
dec_emb = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

attention = Attention()
context_vector = attention([decoder_outputs, encoder_outputs])
decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, context_vector])

decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([q_seqs, a_seqs_in], a_out_oh, batch_size=2, epochs=500, verbose=0)

print("✅ Model trained.")

encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

dec_input = Input(shape=(1,))
dec_state_input_h = Input(shape=(LSTM_UNITS,))
dec_state_input_c = Input(shape=(LSTM_UNITS,))
enc_output_input = Input(shape=(max_q_len, LSTM_UNITS))

dec_emb_inf = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(dec_input)
dec_lstm_out, dec_h, dec_c = decoder_lstm(dec_emb_inf, initial_state=[dec_state_input_h, dec_state_input_c])
attn_inf = Attention()([dec_lstm_out, enc_output_input])
concat_inf = tf.keras.layers.Concatenate(axis=-1)([dec_lstm_out, attn_inf])
dec_out_inf = decoder_dense(concat_inf)

decoder_model = Model(
    [dec_input, dec_state_input_h, dec_state_input_c, enc_output_input],
    [dec_out_inf, dec_h, dec_c]
)

reverse_word_index = {i: w for w, i in tokenizer.word_index.items()}
reverse_word_index[0] = ''

def generate_reply(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_q_len, padding='post')
    enc_out, state_h, state_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq, state_h, state_c, enc_out])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index[sampled_token_index]

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_a_len:
            break
        decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        state_h, state_c = h, c

    return decoded_sentence.strip()

user_input = "how are you"
print(f"User: {user_input}")
print(f"Bot : {generate_reply(user_input)}")
