import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("🚀 กำลังเริ่มเทรนโมเดล...")

# 1. โหลดข้อมูล
df = pd.read_csv('../data/heart.csv')
X = df.drop('target', axis=1)  # Features ทั้งหมด
y = df['target']               # Target (0 หรือ 1)

# 2. แบ่งข้อมูล Train/Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. สร้างและเทรนโมเดล (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. ทดสอบความแม่นยำ
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ ความแม่นยำ (Accuracy): {accuracy:.2%}")
print("\n📋 รายงานผลการทดสอบ:")
print(classification_report(y_test, y_pred))

# 5. บันทึกโมเดลลงโฟลเดอร์ models/
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/heart_model.pkl')
print(" บันทึกโมเดลเสร็จสิ้นที่: models/heart_model.pkl")