import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from train_model import load_dataset, train_model

def evaluate():
    print("Mulai Evaluasi Model Skincare Chatbot...")
    
    # 1. Load Data using existing function (includes augmentation)
    print("Loading Dataset & Performing Augmentation...")
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets.json')
    patterns, labels = load_dataset(dataset_path, augment=True)
    
    print(f"Total Data setelah Augmentasi: {len(patterns)}")
    
    # 2. Split Data (80% Train, 20% Test)
    # Using stratify to ensure class balance
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            patterns, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed (mungkin ada kelas dengan sampel terlalu sedikit). Fallback to random split. Error: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            patterns, labels,
            test_size=0.2,
            random_state=42
        )
    
    print(f"Data Training: {len(X_train)}")
    print(f"Data Testing: {len(X_test)}")
    
    # 3. Train Model
    print("Training Model (Logistic Regression)...")
    # train_model returns (vectorizer, model) and prints accuracy inside
    vectorizer, model = train_model(X_train, X_test, y_train, y_test)
    
    # 4. Detailed Evaluation
    print("Menghitung Detail Metrik...")
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print("\n" + "="*50)
    print(f"HASIL EVALUASI MODEL - Skincare Chatbot")
    print("="*50)
    print(f"Akurasi Model: {accuracy*100:.2f}%")
    print("\nDetail Laporan Klasifikasi:")
    print(report)
    print("="*50)

if __name__ == "__main__":
    evaluate()
