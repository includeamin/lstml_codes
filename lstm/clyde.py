import uuid
from pandas import read_csv
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import statistics
import math

divider_len = 70
space_len = 2

epochs = 64
batch_size = 32

print('_'*divider_len)
print('Reading dataset...')
dataset = read_csv('Clyde.csv').values
print(dataset)
cols = len(dataset[0, :])
rows = len(dataset[:, 0])
print('Size: ', rows, 'x', cols)
print('_'*divider_len + '\n'*space_len)

# print('_'*divider_len)
# print('Plotting target value...')
# plt.plot(list(range(rows)), dataset[:, 0])
# plt.show()
# print('_'*divider_len + '\n'*space_len)

print('_'*divider_len)
print('Spliting data to train and test and validation')
train_data = dataset[:14766, :]
validation_data = train_data[10000:, :]
test_data = dataset[14766:, :]
print('Train data: ', len(train_data[:, 0]))
print('Test data: ', len(test_data[:, 0]))
print('Validation data: ', len(validation_data[:, 0]))
print('_'*divider_len + '\n'*space_len)


print('_'*divider_len)
print('Normalizing data...')
print('Normalizing train data...')
for col in range(len(train_data[0, :])):
    mean_value = statistics.mean(train_data[:, col])
    for row in range(len(train_data[:, col])):
        train_data[row, col] -= mean_value

    if col == 0:
        continue
    max_val = max(train_data[:, col])
    min_val = min(train_data[:, col])
    diff = max_val - min_val
    for row in range(len(train_data[:, col])):
        train_data[row, col] /= diff

print('Normalizing validation data...')
for col in range(len(validation_data[0, :])):
    mean_value = statistics.mean(validation_data[:, col])
    for row in range(len(validation_data[:, col])):
        validation_data[row, col] -= mean_value

    if col == 0:
        continue
    max_val = max(validation_data[:, col])
    min_val = min(validation_data[:, col])
    diff = max_val - min_val
    for row in range(len(validation_data[:, col])):
        validation_data[row, col] /= diff

print('Normalizing test data...')
for col in range(len(test_data[0, :])):
    mean_value = statistics.mean(test_data[:, col])
    for row in range(len(test_data[:, col])):
        test_data[row, col] -= mean_value

    if col == 0:
        continue
    max_val = max(test_data[:, col])
    min_val = min(test_data[:, col])
    diff = max_val - min_val
    for row in range(len(test_data[:, col])):
        test_data[row, col] /= diff
print('_'*divider_len + '\n'*space_len)


print('_'*divider_len)
print('Plotting time series...')
fig = plt.figure()
plt.plot(list(range(0, len(train_data[:, 0]))), train_data, 'b-')
plt.plot(list(range(len(train_data[:, 0]), len(
    validation_data[:, 0]) + len(train_data[:, 0]))), validation_data, 'g-')
plt.plot(list(range(len(train_data[:, 0]) + len(validation_data[:, 0]), len(
    test_data[:, 0]) + len(train_data[:, 0]) + len(validation_data[:, 0]))), test_data, 'r-')
plt.show()
print('_'*divider_len + '\n'*space_len)


print('_'*divider_len)
print('Spliting test and train and validation data to features and target...')
train_X, train_y = train_data[:, 1:], train_data[:, 0]
valid_X, valid_y = validation_data[:, 1:], validation_data[:, 0]
test_X, test_y = test_data[:, 1:], test_data[:, 0]
print('Reshaping data...')
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("Training data shape X, y => ", train_X.shape, train_y.shape, '\n',
      "Testing data shape X, y => ", test_X.shape, test_y.shape, '\n',
      "Valid data shape X, y => ", valid_X.shape, valid_X.shape)
print('_'*divider_len + '\n'*space_len)


print('_'*divider_len)
print('Building model...')
model = Sequential()

model.add(LSTM(128,
               input_shape=(train_X.shape[1], train_X.shape[2]),
               kernel_initializer='random_normal',
               activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))

model.compile(loss='mse', optimizer='adam')
print('Model compiled with:\nOptimizer:{}\nLoss:{}'.format('Adam', 'MSE'))
print('_'*divider_len + '\n'*space_len)

print('_'*divider_len)
print('Training...')
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                    validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
print('Plotting history of training')
plt.plot(history.history['loss'], 'b', label='training history')
plt.plot(history.history['val_loss'],  'r', label='validation history')
plt.title("Train and Validation Loss for the LSTM")
plt.legend()
plt.show()
print('_'*divider_len + '\n'*space_len)

print('_'*divider_len)
print('Predicting...')
yhat = model.predict(test_X)
predictions = []
for hat in yhat:
    predictions.append(hat[0])

loss = 0.0
mloss = 0.0
for i in range(len(predictions)):
    loss += (predictions[i] - test_y[i]) * (predictions[i] - test_y[i])
    mloss += abs(predictions[i] - test_y[i])

print('RMSE:', math.sqrt(loss / len(predictions)))
print('MSE:', loss / len(predictions))
print('MAE:', mloss / len(predictions))

fig = plt.figure()
x = list(range(len(predictions)))
plt.plot(x, test_y, 'b-')
plt.plot(x, predictions, 'g-')
plt.show()
print('_'*divider_len + '\n'*space_len)

print('_'*divider_len)
print('Saving model...')
name = str(uuid.uuid4()) + '.hdf5'
model.save(name)
print('Model saved: ' + name)
print('_'*divider_len + '\n'*space_len)

print('_'*divider_len)
print('Saving csv file...')
name = str('result') + '.csv'
with open(name, 'w') as file_:
	file_.write('target,predict\n')
	for i in range(len(test_y)):
		file_.write(str(test_y[i]) + ',' + str(predictions[i]) + '\n')

print('File saved: ' + name)
print('_'*divider_len + '\n'*space_len)
