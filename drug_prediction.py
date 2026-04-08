# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ========== STEP 1: LOAD DATA ==========
print("="*60)
print("DRUG EFFECTIVENESS PREDICTION PROJECT")
print("="*60)

print("\n[1/6] Loading dataset...")
df = pd.read_csv('drug200.csv')
print(f"✓ Dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")

# Show first few rows
print("\nFirst 5 patients:")
print(df.head())

# ========== STEP 2: EXPLORE DATA ==========
print("\n[2/6] Exploring data...")
print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDrug distribution:")
print(df['Drug'].value_counts())

# ========== STEP 3: PREPROCESS DATA ==========
print("\n[3/6] Preprocessing data...")

# Convert categorical to numerical
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['BP'] = le.fit_transform(df['BP'])
df['Cholesterol'] = le.fit_transform(df['Cholesterol'])
df['Drug'] = le.fit_transform(df['Drug'])

print("✓ Categorical variables encoded")

# Split features and target
X = df.drop('Drug', axis=1)
y = df['Drug']

print(f"✓ Features: {X.columns.tolist()}")
print(f"✓ Target: Drug (5 types)")

# ========== STEP 4: SPLIT DATA ==========
print("\n[4/6] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {len(X_train)} patients")
print(f"✓ Testing set: {len(X_test)} patients")

# ========== STEP 5: TRAIN MODELS ==========
print("\n[5/6] Training models...")

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("✓ Decision Tree trained")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✓ Random Forest trained")

# ========== STEP 6: EVALUATE MODELS ==========
print("\n[6/6] Evaluating models...")
print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Decision Tree results
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"\n🌳 Decision Tree Accuracy: {dt_accuracy:.2%}")

# Random Forest results
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"🌲 Random Forest Accuracy: {rf_accuracy:.2%}")

# Show best model
if rf_accuracy > dt_accuracy:
    best_model = rf_model
    best_name = "Random Forest"
    best_acc = rf_accuracy
    best_pred = rf_pred
else:
    best_model = dt_model
    best_name = "Decision Tree"
    best_acc = dt_accuracy
    best_pred = dt_pred

print(f"\n🏆 Best Model: {best_name} ({best_acc:.2%})")

# Detailed report for best model
print(f"\n📋 Detailed Report for {best_name}:")
print(classification_report(y_test, best_pred))

# ========== BONUS: MAKE PREDICTIONS ==========
print("\n" + "="*60)
print("TESTING PREDICTION SYSTEM")
print("="*60)

# Function to predict for new patients
def predict_drug(age, sex, bp, cholesterol, na_to_k):
    """
    Predict drug for a patient
    sex: 'M' or 'F'
    bp: 'LOW', 'NORMAL', 'HIGH'
    cholesterol: 'NORMAL', 'HIGH'
    """
    # Encode inputs
    sex_enc = 1 if sex == 'M' else 0
    bp_map = {'HIGH': 0, 'LOW': 1, 'NORMAL': 2}
    chol_map = {'HIGH': 0, 'NORMAL': 1}
    
    # Create input
    patient = [[age, sex_enc, bp_map[bp], chol_map[cholesterol], na_to_k]]
    
    # Predict
    prediction = best_model.predict(patient)[0]
    
    # Decode
    drugs = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
    return drugs[prediction]

# Test cases
test_patients = [
    (50, 'M', 'HIGH', 'HIGH', 15.5),
    (25, 'F', 'NORMAL', 'NORMAL', 10.2),
    (35, 'M', 'LOW', 'HIGH', 12.0)
]

print("\n💊 Predictions for test patients:\n")
for i, (age, sex, bp, chol, na_k) in enumerate(test_patients, 1):
    result = predict_drug(age, sex, bp, chol, na_k)
    print(f"Patient {i}: Age={age}, Sex={sex}, BP={bp}, Chol={chol}, Na/K={na_k}")
    print(f"  → Recommended: {result}\n")
    


    

print("="*60)
print("✅ PROJECT COMPLETED!")
print("="*60)

# Visualize feature importance
import matplotlib.pyplot as plt

if best_name == "Random Forest":
    importance = best_model.feature_importances_
    features = X.columns
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\n✓ Saved: feature_importance.png")
    plt.show()


# Save the best model
import pickle

with open('drug_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\n✓ Model saved as 'drug_model.pkl'")
