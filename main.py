import numpy as np
import pandas as pd
from pandarallel import pandarallel
from Preparation import Preparation
from Lstm import Lstm
from CNN_1D import CNN
from RNN import RNN
from TransformerDNN import TransformerDNN
from sklearn.metrics import accuracy_score
import os
import pickle

if __name__ == "__main__":
    pandarallel.initialize(progress_bar=True)

    train_df = pd.read_excel("train.xlsx")

    y_train = train_df['rating']
    test_df = pd.read_csv("test _no_label.csv")
    test_ids = np.array(test_df['ID'])

    x_train = train_df['review_description']

    x_test = test_df['review_description']

    if os.path.exists('train_features.npy') and os.path.exists('test_features.npy'):
        x_train_features = np.load('train_features.npy')
        x_test_features = np.load('test_features.npy')
        with open('tokenizer.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
    else:
        preparation_model = Preparation()
        clean_x_train = x_train.parallel_apply(preparation_model.preprocess)
        x_train_features = preparation_model.feature_extraction_train(clean_x_train)

        clean_x_test = x_test.parallel_apply(preparation_model.preprocess)
        x_test_features, tokenizer = preparation_model.feature_extraction_test(clean_x_test)
        np.save('train_features.npy',x_train_features)
        np.save('test_features.npy', x_test_features)

        with open('tokenizer.pkl', 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)

    # Conv1D model
    cnn_model = CNN(tokenizer)
    train_pred = cnn_model.train(x_train_features,y_train)
    print('CNN Train Accuracy --> ',accuracy_score(y_train,train_pred))
    predictions = cnn_model.test(x_test_features)
    print(predictions)
    result_df_cnn = pd.DataFrame({'ID': test_ids, 'rating': predictions})
    result_df_cnn.to_csv('CNN predictions.csv', index=False)

    # LSTM model
    lstm_model = Lstm(tokenizer)
    train_pred = lstm_model.train(x_train_features, y_train)
    print('Lstm Train Accuracy --> ', accuracy_score(y_train, train_pred))
    predictions = lstm_model.test(x_test_features)
    print(predictions)
    result_df_lstm = pd.DataFrame({'ID': test_ids, 'rating': predictions})
    result_df_lstm.to_csv('LSTM predictions.csv', index=False)

    # RNN model
    rnn_model = RNN(tokenizer)
    train_pred = rnn_model.train(x_train_features, y_train)
    print('RNN Train Accuracy --> ', accuracy_score(y_train, train_pred))
    predictions = rnn_model.test(x_test_features)
    print(predictions)
    result_df_rnn = pd.DataFrame({'ID': test_ids, 'rating': predictions})
    result_df_rnn.to_csv('RNN predictions.csv', index=False)

    # # Transformer DNN model
    # transformer_model = TransformerDNN(tokenizer)
    # train_pred = transformer_model.train(x_train_features,y_train)
    # print('Transformer Train Accuracy --> ',accuracy_score(y_train,train_pred))
    # predictions = transformer_model.test(x_test_features)
    # print(predictions)
    # result_df_lstm = pd.DataFrame({'ID': test_ids, 'rating': predictions})
    # result_df_lstm.to_csv('Transformer predictions.csv', index=False)
