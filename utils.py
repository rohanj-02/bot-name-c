import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
# import Dense and Concatenate from tf
import pickle
from AttentionLayer import AttentionLayer
from tensorflow.keras.layers import Dense, Concatenate


def load_model():
    new_model = tf.keras.models.load_model('model/chatbot.h5')
    return new_model


def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


def predict(input_text):
    VOCAB_SIZE = pickle.loads(open('model/vocab_size.pkl', 'rb').read())
    enc_model = tf.keras.models.load_model(
        'model/encoder_model.h5', compile=False)
    dec_model = tf.keras.models.load_model(
        'model/decoder_model.h5', compile=False)
    vocab = pickle.loads(open('model/vocab.pkl', 'rb').read())
    inv_vocab = pickle.loads(open('model/inv_vocab.pkl', 'rb').read())
    attn_layer = keras.layers.deserialize(pickle.loads(open(
        "model/attn_layer.pkl", "rb").read()), custom_objects={'AttentionLayer': AttentionLayer})
    dec_dense = keras.layers.deserialize(
        pickle.loads(open("model/dec_dense.pkl", "rb").read()))

    prepro1 = input_text.lower()

    try:
        prepro1 = clean_text(prepro1)
        prepro = [prepro1]

        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                try:
                    lst.append(vocab[y])
                except:
                    lst.append(vocab['<OUT>'])
            txt.append(lst)
        txt = pad_sequences(txt, 15, padding='post')

        print(txt.shape)
        enc_op, stat = enc_model.predict(txt)
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = vocab['<SOS>']
        stop_condition = False
        decoded_translation = ''
        # print(stat[0])
        typw = [empty_target_seq] + stat
        print(len(typw))
        for r in typw:
            print(r.shape)

        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + stat)
            # print(dec_outputs.shape, h.shape, c.shape)
            attn_op, attn_state = attn_layer([enc_op, dec_outputs])
            # print(attn_op.shape, attn_state.shape)
            decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])
            # print(decoder_concat_input.shape)
            decoder_concat_input = dec_dense(decoder_concat_input)
            sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
            sampled_word = inv_vocab[sampled_word_index] + ' '
            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word
            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 20:
                stop_condition = True
            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            stat = [h, c]

        # print("chatbot attention : ", decoded_translation)
        return decoded_translation
        # print("==============================================")

    except:
        return "Sorry! I did not understand. Please try again."


if __name__ == "__main__":
    predict("Helloo")
    # inp = ""

    # while inp != ":q":
    #     inp = input("you : ")
    #     if inp != ":q":
    #         predict(inp)
