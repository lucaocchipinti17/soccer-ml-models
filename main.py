from data.scraper import DataScraper
import numpy as np
import matplotlib.pyplot as plt
from models.nn_diff import NeuralNetworkDiff
from models.nn_winner import NeuralNetworkWinner
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

leagues = ['EPL', 'La_Liga', 'Serie_A', 'Bundesliga', 'Ligue_1']
seasons = [str(year) for year in range(2014, 2025)]

d = DataScraper(leagues=leagues, seasons=seasons)
d.get_data()

def train_diff_model():
    X_train, y_train, X_valid, y_valid = d.prepare_for_training_diff()

    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)

    model = NeuralNetworkDiff(input_size=X_train.shape[1], 
    hidden_size=32, hidden_size2=32, learning_rate=1e-3, huber_delta=0.5)
    model.train(X_train, y_train, epochs=1000, verbose=True)

    y_pred = model.predict(X_valid)
    val_loss = model.compute_loss(y_valid, y_pred)
    mae = model.compute_mae(y_valid, y_pred)
    mse = model.compute_mse(y_valid, y_pred)
    print("Validation MAE:", mae)
    print("Validation Huber Loss:", val_loss)
    print("Validation MSE:", mse)

    # Baseline: always predict 0 goal differential
    y_pred_zero = np.zeros_like(y_valid)
    baseline_mae = model.compute_mae(y_valid, y_pred_zero)
    baseline_mse = model.compute_mse(y_valid, y_pred_zero)
    baseline_loss = model.compute_loss(y_valid, y_pred_zero)
    print("Baseline (predict 0) MAE:", baseline_mae)
    print("Baseline (predict 0) MSE:", baseline_mse)
    print("Baseline (predict 0) Loss:", baseline_loss)

    # Plot: Predicted vs Actual goal differential
    y_true = y_valid.ravel()
    y_hat = y_pred.ravel()
    lim = max(1.0, float(np.max(np.abs(np.concatenate([y_true, y_hat], axis=0)))))
    lim = min(lim, 6.0)
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_hat, alpha=0.4, s=10)
    plt.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
    plt.xlabel('Actual goal differential')
    plt.ylabel('Predicted goal differential')
    plt.title('Predicted vs Actual (Goal Differential)')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

def train_winner_model():
    X_train, y_train, X_valid, y_valid = d.prepare_for_training_winner()

    model = NeuralNetworkWinner(input_size=X_train.shape[1], 
    hidden_size=32, hidden_size2=32, learning_rate=1e-4, weight_decay=5e-4)
    model.train(X_train, y_train, epochs=1000, verbose=True)
    probs = model.predict_proba(X_valid)
    val_loss = model.compute_loss(y_valid, probs)
    accuracy = model.compute_accuracy(y_valid, probs)
    print("Validation Accuracy:", accuracy)
    print("Validation Loss:", val_loss)

def train_xgboost_winner():
    """Train XGBoost to predict match winner."""
    X_train, y_train_onehot, X_valid, y_valid_onehot = d.prepare_for_training_winner()
    
    # Convert one-hot to class indices for XGBoost (0=home_win, 1=draw, 2=away_win)
    y_train_classes = np.argmax(y_train_onehot, axis=1)
    y_valid_classes = np.argmax(y_valid_onehot, axis=1)
    
    # XGBoost classifier for multi-class classification
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=1
    )
    
    print("Training XGBoost model...")
    xgb_model.fit(
        X_train, y_train_classes,
        eval_set=[(X_valid, y_valid_classes)],
        verbose=50
    )
    
    # Predictions
    y_pred_classes = xgb_model.predict(X_valid)
    y_pred_probs = xgb_model.predict_proba(X_valid)
    
    # Convert back to one-hot for compatibility
    y_pred_onehot = np.zeros_like(y_valid_onehot)
    y_pred_onehot[np.arange(len(y_pred_classes)), y_pred_classes] = 1.0
    
    # Accuracy
    accuracy = accuracy_score(y_valid_classes, y_pred_classes)
    print(f"\nXGBoost Validation Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = ['Home Win', 'Draw', 'Away Win']
    print("\nClassification Report:")
    print(classification_report(y_valid_classes, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_valid_classes, y_pred_classes)
    print(cm)
    
    return xgb_model

def compare_winner_models():
    """Train and compare Neural Network vs XGBoost for winner prediction."""
    print("=" * 70)
    print("COMPARING WINNER PREDICTION MODELS")
    print("=" * 70)
    
    # Prepare data once for both models
    X_train, y_train_onehot, X_valid, y_valid_onehot = d.prepare_for_training_winner()
    y_valid_classes = np.argmax(y_valid_onehot, axis=1)
    
    print("\n" + "-" * 70)
    print("NEURAL NETWORK")
    print("-" * 70)
    
    # Train Neural Network
    nn_model = NeuralNetworkWinner(
        input_size=X_train.shape[1], 
        hidden_size=32, 
        hidden_size2=32, 
        learning_rate=1e-4, 
        weight_decay=5e-4
    )
    nn_model.train(X_train, y_train_onehot, epochs=1000, verbose=True)
    
    # Evaluate Neural Network
    nn_probs = nn_model.predict_proba(X_valid)
    nn_pred_classes = np.argmax(nn_probs, axis=1)
    nn_accuracy = nn_model.compute_accuracy(y_valid_onehot, nn_probs)
    nn_loss = nn_model.compute_loss(y_valid_onehot, nn_probs)
    
    print(f"\nNeural Network Results:")
    print(f"  Accuracy: {nn_accuracy:.4f}")
    print(f"  Loss (Cross-Entropy): {nn_loss:.4f}")
    
    print("\n" + "-" * 70)
    print("XGBOOST")
    print("-" * 70)
    
    # Train XGBoost
    y_train_classes = np.argmax(y_train_onehot, axis=1)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=1
    )
    
    print("Training XGBoost model...")
    xgb_model.fit(
        X_train, y_train_classes,
        eval_set=[(X_valid, y_valid_classes)],
        verbose=50
    )
    
    # Evaluate XGBoost
    xgb_pred_classes = xgb_model.predict(X_valid)
    xgb_probs = xgb_model.predict_proba(X_valid)
    xgb_accuracy = accuracy_score(y_valid_classes, xgb_pred_classes)
    
    # Calculate cross-entropy loss for XGBoost
    xgb_pred_onehot = np.zeros_like(y_valid_onehot)
    xgb_pred_onehot[np.arange(len(xgb_pred_classes)), xgb_pred_classes] = 1.0
    eps = 1e-12
    xgb_probs_clipped = np.clip(xgb_probs, eps, 1.0)
    xgb_loss = -np.mean(np.sum(y_valid_onehot * np.log(xgb_probs_clipped), axis=1))
    
    print(f"\nXGBoost Results:")
    print(f"  Accuracy: {xgb_accuracy:.4f}")
    print(f"  Loss (Cross-Entropy): {xgb_loss:.4f}")
    
    # Side-by-side comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'Neural Network':<20} {'XGBoost':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<30} {nn_accuracy:<20.4f} {xgb_accuracy:<20.4f}")
    print(f"{'Cross-Entropy Loss':<30} {nn_loss:<20.4f} {xgb_loss:<20.4f}")
    print("=" * 70)
    
    # Per-class performance
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    print("\nNEURAL NETWORK - Per-class Performance:")
    print(classification_report(y_valid_classes, nn_pred_classes, target_names=class_names, digits=4))
    
    print("\nXGBOOST - Per-class Performance:")
    print(classification_report(y_valid_classes, xgb_pred_classes, target_names=class_names, digits=4))
    
    print("\nNEURAL NETWORK - Confusion Matrix:")
    print(confusion_matrix(y_valid_classes, nn_pred_classes))
    
    print("\nXGBOOST - Confusion Matrix:")
    print(confusion_matrix(y_valid_classes, xgb_pred_classes))
    
    return nn_model, xgb_model

if __name__ == "__main__":
    compare_winner_models()
