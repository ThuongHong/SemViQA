import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc file dữ liệu gốc
file_path = "vihallu-train.csv"  # Đường dẫn tới file gốc
data = pd.read_csv(file_path)

# 1. Đổi tên cột
data.rename(columns={
    "context": "evidence",
    "response": "claim",
    "label": "verdict"
}, inplace=True)

# 2. Mapping giá trị ở cột 'verdict'
mapping = {
    "no": "SUPPORTED",
    "intrinsic": "REFUTED",
    "extrinsic": "NEI"
}
data["verdict"] = data["verdict"].map(mapping)

# 3. Split thành train và test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Lưu thành file CSV
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

print("Đã hoàn thành việc xử lý và chia dữ liệu!")