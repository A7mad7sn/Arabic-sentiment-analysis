import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, MultiHeadAttention, Dense,Flatten
from keras.models import save_model, load_model
from keras.utils import to_categorical
import os


class TransformerDNN:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model = Sequential()

    def assign_class(self, output_list):
        new_output_list = []
        for node_output in output_list:
            small_list = []
            for digit in node_output:
                if digit == max(node_output):
                    small_list.append(1)
                else:
                    small_list.append(0)
            new_output_list.append(small_list)

        new_output_list = np.array(new_output_list)
        actual_predictions = []
        for small_list in new_output_list:
            if np.array_equal(small_list, np.array([1, 0, 0])):
                actual_predictions.append(0)
            if np.array_equal(small_list, np.array([0, 1, 0])):
                actual_predictions.append(1)
            if np.array_equal(small_list, np.array([0, 0, 1])):
                actual_predictions.append(-1)
        return actual_predictions

    def train(self, x_train_features, y_train):
        y_train = to_categorical(y_train, num_classes=3)
        if os.path.exists('transformer_model.h5'):
            self.model = load_model('transformer_model.h5')
        else:

            self.model = Sequential()
            self.model.add(Embedding(input_dim=max(self.tokenizer.index_word.keys())+1, output_dim=200, input_length=x_train_features.shape[1]))
            self.model.add(Flatten())
            self.model.add(MultiHeadAttention(num_heads=8, key_dim=200))
            self.model.add(Dense(units=3, activation='softmax'))

            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(x_train_features, y_train, epochs=1)

            save_model(self.model, 'transformer_model.h5')

        # Predictions
        predictions = self.model.predict(x_train_features)
        predictions = self.assign_class(predictions)

        return predictions

    def test(self, x_test_features):
        predictions = self.model.predict(x_test_features)
        predictions = self.assign_class(predictions)

        return predictions