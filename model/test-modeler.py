import xgboost as xgb
import numpy as np

# Create a dummy dataset
X = np.random.rand(100, 4)  # 100 samples, 4 features
y = np.random.randint(0, 2, 100)  # Random labels

# Train a dummy model
model = xgb.XGBClassifier()
model.fit(X, y)

# Save it
model.save_model("ipda_model.json")
print("Dummy model saved to ipda_model.json")