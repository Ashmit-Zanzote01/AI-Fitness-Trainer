import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Layer aliases
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
TensorBoard = tf.keras.callbacks.TensorBoard
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
Sequential = tf.keras.models.Sequential
Input = tf.keras.layers.Input
l2 = tf.keras.regularizers.l2

# Optimizers
Adam = tf.keras.optimizers.Adam
RMSprop = tf.keras.optimizers.RMSprop
SGD = tf.keras.optimizers.SGD

# Paths
MODEL_DIR = "D:/final_year_project/models"
LOG_DIR = "D:/final_year_project/logs"
SCALER_PATH = "D:/final_year_project/scaler.pkl"
EVAL_DIR = "D:/final_year_project/evaluation"
TRAIN_DATA_PATH = "D:/final_year_project/Final_Dataset/Extracted_Features"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def load_data():
    print("🔄 Loading extracted feature data...")
    X, y = [], []
    label_map = {}
    label_counter = 0

    for exercise in sorted(os.listdir(TRAIN_DATA_PATH)):
        exercise_path = os.path.join(TRAIN_DATA_PATH, exercise)
        if not os.path.isdir(exercise_path):
            continue

        if exercise not in label_map:
            label_map[exercise] = label_counter
            label_counter += 1

        for file in os.listdir(exercise_path):
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(exercise_path, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            frames = data["data"]
            if len(frames) < 30:
                continue

            # Process sequences with 25% overlap
            for i in range(0, len(frames) - 30 + 1, 30 // 4):
                sequence = frames[i:i+30]
                
                # Extract all features with explicit dtype conversion
                angles = np.array([frame["angles"] for frame in sequence], dtype=np.float32)
                distances = np.array([frame["distances"] for frame in sequence], dtype=np.float32)
                y_distances = np.array([frame["y_distances"] for frame in sequence], dtype=np.float32)
                z_features = np.array([frame["z_features"] for frame in sequence], dtype=np.float32)

                # Add noise to all features
                angles += np.random.normal(0, 0.05, angles.shape).astype(np.float32)
                distances += np.random.normal(0, 0.05, distances.shape).astype(np.float32)
                y_distances += np.random.normal(0, 0.05, y_distances.shape).astype(np.float32)
                z_features += np.random.normal(0, 0.05, z_features.shape).astype(np.float32)
                
                # Synthetic depth variations augmentation
                z_features *= np.random.uniform(0.9, 1.1)  # Added depth variation

                # Combine all features (30 total: 8+14+4+4)
                feature_sequence = np.hstack([
                    angles, 
                    distances, 
                    y_distances,
                    z_features
                ])
                
                X.append(feature_sequence)
                y.append(label_map[exercise])

    return np.array(X), np.array(y), label_map

# Load and split data
X, y, label_map = load_data()
print(f"✅ Loaded {len(X)} sequences across {len(label_map)} exercise classes.")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.111, stratify=y_train_full, random_state=42
)
print(f"✅ Data Split: {len(X_train)} Train | {len(X_val)} Validation | {len(X_test)} Test")

# Feature scaling
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 30)  # 30 features
X_val_flat = X_val.reshape(-1, 30)
X_test_flat = X_test.reshape(-1, 30)

scaler.fit(X_train_flat)

X_train = scaler.transform(X_train_flat).reshape(-1, 30, 30)
X_val = scaler.transform(X_val_flat).reshape(-1, 30, 30)
X_test = scaler.transform(X_test_flat).reshape(-1, 30, 30)

# Class weighting
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Model architecture
def build_model(optimizer):
    model = Sequential([
        Input(shape=(30, 30)),  # 30 timesteps, 30 features
        Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.5),
        Bidirectional(LSTM(16, return_sequences=False, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(label_map), activation='softmax')
    ])
    model.compile(optimizer=optimizer, 
                loss="sparse_categorical_crossentropy", 
                metrics=["accuracy"])
    return model

# Optimizers
optimizers = {
    "Adam": Adam(learning_rate=0.0005),
    "RMSprop": RMSprop(learning_rate=0.0005),
    "SGD-Nesterov": SGD(learning_rate=0.00025, momentum=0.9, nesterov=True)
}

# Training loop
history_dict = {}
best_optimizer = None
best_val_acc = 0

for opt_name, optimizer in optimizers.items():
    print(f"\n🔄 Training with {opt_name} optimizer...")
    
    model = build_model(optimizer)
    checkpoint_path = os.path.join(MODEL_DIR, f"best_model_{opt_name}.keras")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
            TensorBoard(log_dir=LOG_DIR),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        ],
        verbose=1
    )
    
    history_dict[opt_name] = history.history
    max_val_acc = max(history.history['val_accuracy'])
    
    if max_val_acc > best_val_acc:
        best_val_acc = max_val_acc
        best_optimizer = opt_name
        
    print(f"✅ {opt_name} - Best Validation Accuracy: {max_val_acc:.4f}")

# Final evaluation
print("\n🚀 Final Evaluation on Test Set...")
final_model = build_model(optimizers[best_optimizer])
final_model.load_weights(os.path.join(MODEL_DIR, f"best_model_{best_optimizer}.keras"))

test_loss, test_acc = final_model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Generate predictions
y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
class_names = list(label_map.keys())
report = classification_report(y_test, y_pred_classes, target_names=class_names)
print("\nClassification Report:")
print(report)

# Save classification report
with open(os.path.join(EVAL_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'))
plt.close()

# Optimizer comparison plot
optimizer_names = list(history_dict.keys())
best_val_accs = [max(hist['val_accuracy']) for hist in history_dict.values()]

plt.figure(figsize=(10, 6))
plt.bar(optimizer_names, best_val_accs, color=['#4CAF50', '#2196F3', '#FF9800'])
plt.ylabel('Validation Accuracy')
plt.title('Optimizer Performance Comparison')
plt.ylim(0, 1)
plt.savefig(os.path.join(EVAL_DIR, 'optimizer_comparison.png'))
plt.close()

# Training history visualization
plt.figure(figsize=(12, 6))
for opt_name, hist in history_dict.items():
    plt.plot(hist['accuracy'], label=f'{opt_name} Train', linestyle='--')
    plt.plot(hist['val_accuracy'], label=f'{opt_name} Val')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(EVAL_DIR, 'training_history.png'))
plt.close()

# Save artifacts
final_model.save(os.path.join(MODEL_DIR, "final_exercise_model_3d.keras"))
joblib.dump(scaler, SCALER_PATH)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder_3d.pkl"))

print("✅ Training completed and all artifacts saved!")