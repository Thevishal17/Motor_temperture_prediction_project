import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the dataset
df = pd.read_csv("data/motor_temperature.csv")
print("\nInitial Data Preview:")
print(df.head())

# 2. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Fill missing values if any
df.fillna(df.mean(), inplace=True)

# 4. Visualize correlation (optional, commented to avoid blocking)
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title("Feature Correlation Heatmap")
# plt.show()

# 5. Split features and target
X = df.drop("motor_temp", axis=1)
y = df["motor_temp"]

# 6. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, "models/scaler.pkl")

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Save split data for training script
np.save("models/X_train.npy", X_train)
np.save("models/X_test.npy", X_test)
np.save("models/y_train.npy", y_train)
np.save("models/y_test.npy", y_test)

print("\nPreprocessing complete. Scaled data and scaler saved.")
