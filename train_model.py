import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. โหลดข้อมูลจากโฟลเดอร์ data/
df = pd.read_csv('../data/heart.csv')

print(f"📊 จำนวนข้อมูล: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
print(f"🎯 สัดส่วนคลาส:\n{df['target'].value_counts()}")

# 2. แยก Features (X) และ Target (y)
X = df.drop('target', axis=1)
y = df['target']

# 3. แบ่งข้อมูล Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. สร้างและเทรนโมเดล (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. ทดสอบโมเดล
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.2%}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# 6. บันทึกโมเดล (สร้างโฟลเดอร์ models/ ถ้ายังไม่มี)
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/heart_model.pkl')
print("\n💾 บันทึกโมเดลที่: models/heart_model.pkl")

# 7. บันทึกชื่อ Features ไว้ใช้ตอนทำ Web App
joblib.dump(X.columns.tolist(), '../models/feature_names.pkl')
print("💾 บันทึกชื่อ Features ที่: models/feature_names.pkl")