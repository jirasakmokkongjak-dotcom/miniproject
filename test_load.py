import pandas as pd

# โหลดไฟล์จากโฟลเดอร์ data/
try:
    df = pd.read_csv('data/heart.csv')
    print("✅ โหลดไฟล์สำเร็จ!")
    print(f"📊 ขนาดข้อมูล: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    print("\n📋 ตัวอย่างข้อมูล 5 แถวแรก:")
    print(df.head())
    print("\n🎯 สัดส่วนคลาส (Target):")
    print(df['target'].value_counts())
except FileNotFoundError:
    print("❌ ไม่พบไฟล์! กรุณาตรวจสอบว่าไฟล์ heart.csv อยู่ในโฟลเดอร์ data/ หรือไม่")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")