import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load preprocessed data
X_train = np.load("models/X_train.npy")
X_test = np.load("models/X_test.npy")
y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # Compute MSE
rmse = np.sqrt(mse)  # Manually compute RMSE (compatible with all sklearn versions)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:\nRMSE: {rmse:.2f}\nR^2 Score: {r2:.2f}")

# Save the trained model
joblib.dump(model, "models/model.pkl")
print("\nModel has been saved as 'models/model.pkl'.")
