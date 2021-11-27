import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from tensorflow.keras.layers import TextVectorization
from transformer import TransformerDecoder, TransformerEncoder, PositionalEmbedding


def get_vectorizer(filename):
    from_disk = pickle.load(open(filename, "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])
    return new_v


def predict(input_sentence):
    # load transformer model
    custom_objects = {"PositionalEmbedding": PositionalEmbedding,
                      "TransformerEncoder": TransformerEncoder, "TransformerDecoder": TransformerDecoder}

    # with keras.utils.custom_object_scope(custom_objects):
    transformer = tf.keras.models.load_model('model/new_transformer/transformer.h5',
                                             custom_objects=custom_objects)

    ques_vectorization = get_vectorizer(
        'model/new_transformer/ques_vectorization.pkl')
    ans_vectorization = get_vectorizer(
        'model/new_transformer/ans_vectorization.pkl')
    ans_index_lookup = pickle.load(
        open('model/new_transformer/ans_index_lookup.pkl', 'rb'))

    max_decoded_sentence_length = 20
    tokenized_input_sentence = ques_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = ans_vectorization([decoded_sentence])[
            :, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = ans_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break

    return decoded_sentence


if __name__ == "__main__":
    # ques_vectorization = get_vectorizer('model/new_transformer/ques_vectorization.pkl')
    # ans_vectorization = get_vectorizer('model/new_transformer/ans_vectorization.pkl')
    # print(ques_vectorization(["How are you?"]))
    # print(ans_vectorization(["I am good"]))
    predict("How are you?")
    # inp = ""
    # while inp != ":q":
    #     inp = input("you : ")
    #     if inp != ":q":
    #         predict(inp)
