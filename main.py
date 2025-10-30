from data.scraper import DataScraper
from models.nn_diff import NeuralNetwork

leagues = ['EPL', 'La_Liga', 'Serie_A', 'Bundesliga', 'Ligue_1']
seasons = [str(year) for year in range(2014, 2025)]

d = DataScraper(leagues=leagues, seasons=seasons)
d.get_data()
X_train, y_train, X_valid, y_valid = d.prepare_for_training()

y_train = y_train.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)

model = NeuralNetwork(input_size=X_train.shape[1], 
hidden_size=32, hidden_size2=16, learning_rate=0.01, huber_delta=1.0)
model.train(X_train, y_train, epochs=1000, verbose=True)

y_pred = model.predict(X_valid)
val_loss = model.compute_loss(y_valid, y_pred)
mae = model.compute_mae(y_valid, y_pred)
print("Validation MAE:", mae)
print("Validation MSE:", val_loss)

