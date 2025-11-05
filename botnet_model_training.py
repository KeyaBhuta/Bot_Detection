# Botnet Detection - Machine Learning Model Training
# Save this as: botnet_model_training.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("BOTNET DETECTION - MODEL TRAINING")
print("=" * 60)

# ========================================
# STEP 1: LOAD PREPROCESSED DATA
# ========================================
print("\n[STEP 1] Loading Preprocessed Data...")

try:
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"✓ Training data loaded: {X_train.shape}")
    print(f"✓ Test data loaded: {X_test.shape}")
    print(f"✓ Number of features: {len(feature_names)}")
    print(f"✓ Training samples: {len(y_train)}")
    print(f"✓ Test samples: {len(y_test)}")
    
except FileNotFoundError:
    print("✗ Error: Preprocessed data not found!")
    print("Please run botnet_preprocessing.ipynb first to generate the data files")
    exit(1)

# Check class distribution
print(f"\nTraining set distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} ({count/len(y_train)*100:.2f}%)")

# ========================================
# STEP 2: TRAIN MULTIPLE MODELS
# ========================================
print("\n[STEP 2] Training Multiple ML Models...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        random_state=42
    )
}

results = {}

print("\nTraining models...\n")
for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred_test,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  ✓ Train Accuracy: {train_acc*100:.2f}%")
    print(f"  ✓ Test Accuracy: {test_acc*100:.2f}%")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ ROC-AUC: {roc_auc:.4f}\n")

# ========================================
# STEP 3: SELECT BEST MODEL
# ========================================
print("\n[STEP 3] Model Comparison...")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [r['train_accuracy'] for r in results.values()],
    'Test Accuracy': [r['test_accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1_score'] for r in results.values()],
    'ROC-AUC': [r['roc_auc'] for r in results.values()]
})

print("\n" + comparison_df.to_string(index=False))

# Select best model based on F1-Score
best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
best_model = results[best_model_name]['model']

print(f"\n✓ Best Model Selected: {best_model_name}")
print(f"  Test Accuracy: {results[best_model_name]['test_accuracy']*100:.2f}%")
print(f"  F1-Score: {results[best_model_name]['f1_score']:.4f}")

# ========================================
# STEP 4: DETAILED EVALUATION
# ========================================
print("\n[STEP 4] Detailed Model Evaluation...")

y_pred = results[best_model_name]['y_pred']
y_pred_proba = results[best_model_name]['y_pred_proba']

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Botnet']))

# ========================================
# STEP 5: VISUALIZATIONS
# ========================================
print("\n[STEP 5] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Comparison
ax = axes[0, 0]
x = np.arange(len(comparison_df))
width = 0.35
ax.bar(x - width/2, comparison_df['Train Accuracy'], width, label='Train Accuracy', alpha=0.8)
ax.bar(x + width/2, comparison_df['Test Accuracy'], width, label='Test Accuracy', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. Confusion Matrix Heatmap
ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Botnet'],
            yticklabels=['Normal', 'Botnet'])
ax.set_title(f'Confusion Matrix - {best_model_name}')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# 3. ROC Curve
ax = axes[1, 0]
for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
ax.grid(alpha=0.3)

# 4. Feature Importance (for Random Forest)
ax = axes[1, 1]
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15 features
    ax.barh(range(len(indices)), importances[indices], alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 15 Feature Importances - {best_model_name}')
    ax.grid(axis='x', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Feature importance not available\nfor this model type',
            ha='center', va='center', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'model_evaluation.png'")

# ========================================
# STEP 6: SAVE BEST MODEL
# ========================================
print("\n[STEP 6] Saving Model and Artifacts...")

# Save the best model
with open('botnet_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Model saved as 'botnet_model.pkl'")

# Save model information
model_info = {
    'model_type': best_model_name,
    'train_accuracy': results[best_model_name]['train_accuracy'],
    'test_accuracy': results[best_model_name]['test_accuracy'],
    'precision': results[best_model_name]['precision'],
    'recall': results[best_model_name]['recall'],
    'f1_score': results[best_model_name]['f1_score'],
    'roc_auc': results[best_model_name]['roc_auc'],
    'n_features': len(feature_names),
    'training_samples': len(y_train),
    'test_samples': len(y_test)
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Model info saved as 'model_info.pkl'")

# ========================================
# STEP 7: GENERATE SAMPLE TEST DATA
# ========================================
print("\n[STEP 7] Generating Sample Test Data...")

# Create a small sample CSV for testing the Flask app
sample_size = 100
sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)

# Get original feature values (before scaling)
# We'll create synthetic data based on test set statistics
sample_data = pd.DataFrame(X_test[sample_indices], columns=feature_names)

# Save sample data
sample_data.to_csv('sample_test_data.csv1', index=False)
print(f"✓ Sample test data saved as 'sample_test_data.csv' ({sample_size} samples)")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated Files:")
print("  ✓ botnet_model.pkl - Trained ML model")
print("  ✓ model_info.pkl - Model performance metrics")
print("  ✓ model_evaluation.png - Visualization charts")
print("  ✓ sample_test_data.csv - Sample data for testing")
print("\nNext Steps:")
print("  1. Run the Flask application: python app.py")
print("  2. Open browser: http://localhost:5000")
print("  3. Upload 'sample_test_data.csv' to test the system")
print("=" * 60)