import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():
    data = pd.read_csv("AAPL.csv", date_parser=True)
    # print(data.tail())
    training_data = data[data["Date"] < "2019-03-24"].copy()
    test_data = data[data["Date"] > "2019-03-24"].copy()

    training_data = training_data.drop(["Date", "Adj Close"], axis=1)

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    # print(training_data)
    # print("---")
    # print(training_data)
    print(training_data.shape[0])
    print(test_data.shape[0])
    X_train = []  # first 60 days of data
    y_train = []
    for i in range(60, training_data.shape[0]):
        X_train.append(training_data[i - 60 : i])  # day 0 - 60 etc
        print("i " + str(i))
        X_train
        y_train.append(training_data[i, 0])  # day 61
        y_train
    X_train = np.array(X_train)
    print(X_train.shape)
    y_train = np.array(y_train)

    regressior = Sequential()

    regressior.add(
        LSTM(
            units=60,
            activation="relu",
            return_sequences=True,
            input_shape=(X_train.shape[1], 5),
        )
    )
    regressior.add(Dropout(0.2))

    regressior.add(LSTM(units=60, activation="relu", return_sequences=True))
    regressior.add(Dropout(0.3))

    regressior.add(LSTM(units=80, activation="relu", return_sequences=True))
    regressior.add(Dropout(0.4))

    regressior.add(LSTM(units=120, activation="relu"))
    regressior.add(Dropout(0.5))

    regressior.add(Dense(units=1))

    print(regressior.summary())

    regressior.compile(optimizer="adam", loss="mean_squared_error")
    regressior.fit(X_train, y_train, epochs=1, batch_size=32)

    training_data = data[data["Date"] < "2019-03-24"].copy()
    test_data = data[data["Date"] > "2019-03-24"].copy()

    last_60_days = training_data.tail(60)
    df = last_60_days.append(test_data, ignore_index=True)
    df = df.drop(["Date", "Adj Close"], axis=1)

    inputs = scaler.transform(df)

    X_test = []
    y_test = []

    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60 : i])
        y_test.append(inputs[i, 0])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print("shapes----------------------------")
    print(X_test.shape)
    print(y_test.shape)
    print("----------------------------")
    print("x_test----------------------------")
    print(X_test)
    print("----------------------------")
    print("predict----------------------------")
    print(regressior.predict(X_test))
    print("----------------------------")
    y_pred = regressior.predict(X_test, verbose=1)
    print(scaler.scale_[0])

    print("y_pred----------------------------")
    print(y_pred)
    print("----------------------------")
    y_pred = y_pred * scaler.scale_[0]
    y_test = y_test * scaler.scale_[0]

    # Visualising the results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color="red", label="Actual Stock Price")
    plt.plot(y_pred, color="blue", label="Predicted Stock Price")
    plt.title("Apple Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
