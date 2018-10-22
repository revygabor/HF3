import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#TODO:earlystopping, vagy mas callback

def create_and_train_model(input_shape, x, y, model_name):
    model = Sequential()
    model.add(Dense(input_shape=[input_shape], units=256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(units=64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(units=32))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')


    # defining callbacks:
    # early stopping callback:
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)

    # model chackpointer
    model_checkpoint = ModelCheckpoint('data/{}'.format(model_name))

    model.fit(x, y, batch_size=16, epochs=100, validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

    model = load_model('data/{}'.format(model_name)) # load the latest saved version
    return model

def preprocess_data_for_n_days_ahead(n_days, train_split):
    df = pd.read_csv('data/weather_dataset.csv') # reads the csv file
    data = np.array(df) # convert it ino numpy ndarray

    x = data[:-n_days, 1:4].astype(float) # selecting columns(tmin, tmax, wind) and selecting days
    y = ((data[n_days:, 1] + data[n_days:, 2]) / 2).astype(float) # selecting days shifted from x by n_days

    data_size = x.shape[0] # number of datas in the dataset
    shuffle_index = np.arange(data_size)
    np.random.shuffle(shuffle_index) # we shuffle the x and y the same way using shuffle indices
    x = x[shuffle_index] # shuffle x data
    y = y[shuffle_index] # shuffle y data

    x_train, y_train = x[:int(data_size*train_split)], y[:int(data_size*train_split)] # train(+validation) split
    x_test,  y_test  = x[int(data_size*train_split):], y[int(data_size*train_split):] # test split

    ssc = StandardScaler().fit(x_train) # fit standard scaler only to training data
    x_train = ssc.transform(x_train) # scale traininng data
    x_test = ssc.transform(x_test) # scale test data

    # fit minmax scaler to training data
    mmsc = MinMaxScaler(feature_range=(-1,1)).fit(y_train[..., None]) # the scaler sccpets only 2D arrays
    # so we add one axis to the end and the reshape it back
    y_train = mmsc.transform(y_train.reshape(-1, 1)).reshape(-1) # minmax scale training data to (-1,1)
    y_test = mmsc.transform(y_test.reshape(-1, 1)).reshape(-1) # minmax scale test data with the same settings

    return (x_train, y_train), (x_test, y_test), ssc, mmsc


if __name__ == '__main__':
    n_days = 1
    (x_train, y_train), (x_test, y_test), ssc, mmsc = preprocess_data_for_n_days_ahead(n_days, 0.8)
    input_shape=x_train[0].shape[0]
    model_name = 'model_{}_days'.format(n_days)
    model = create_and_train_model(input_shape=input_shape, x=x_train, y=y_train, model_name=model_name)

    predictions = model.predict(x_test)
    predictions = mmsc.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
    y_test = mmsc.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    for pred, y in zip(predictions, y_test):
        print('{} {}'.format(pred, y))


    pickle.dump(ssc, open('data/sscaler_{}_days.sav'.format(n_days), 'wb'))
    pickle.dump(mmsc, open('data/mmscaler_{}_days.sav'.format(n_days), 'wb'))

