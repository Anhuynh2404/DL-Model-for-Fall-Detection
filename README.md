# Fall Detection System using MLP + CNN

## 📌 Giới thiệu
Hệ thống phát hiện té ngã sử dụng mô hình kết hợp **MLP (Multi-Layer Perceptron) + CNN (Convolutional Neural Network)**. Mô hình này được huấn luyện trên **SisFall Dataset** để nhận diện các hành động té ngã và hoạt động bình thường, giúp hỗ trợ giám sát sức khỏe cho người cao tuổi hoặc người có nguy cơ té ngã.

## 📂 Dataset
Hệ thống sử dụng **SisFall Dataset**, bao gồm dữ liệu gia tốc và con quay hồi chuyển được ghi lại từ các thiết bị đeo.

- **SisFall Dataset**: [Tải tại đây](https://www.kaggle.com/datasets/kushajmallick/sisfalldataset)
- **SisFall Enhanced Dataset (có nhãn)**: [Tải tại đây](https://www.kaggle.com/datasets/nvnikhil0001/sisfall-enhanced)
- Hoặc có thể tải bằng lệnh:
  ```bash
  !gdown -q 1-E-TLd5_J-DDWZXkuYL-moMpoezlMn4Z  # SisFall Dataset
  !gdown -q 1gvOuxPc8dNgTnxuvPcVuCKifOf98-TV0  # SisFall Enhanced Dataset (Labels)
  ```

## ⚙️ Cài đặt
### Yêu cầu hệ thống
- Python >= 3.7
- TensorFlow >= 2.0
- NumPy, Pandas, Scikit-learn, Matplotlib

### Cài đặt thư viện
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow tqdm gdown
```

## 🚀 Triển khai
### 1️⃣ Tiền xử lý dữ liệu
```python
from dataset_processor import DatasetProcessor

# Khởi tạo bộ xử lý dữ liệu
dp = DatasetProcessor()

# Lấy danh sách file train và test
train_files, test_files = dp.get_file_name("/path/to/dataset/")

# Chuyển dữ liệu sang numpy array
train_data = dp.datasets_to_nparray(train_files)
test_data = dp.datasets_to_nparray(test_files)
```

### 2️⃣ Tạo mô hình MLP + CNN
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(200, 9)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### 3️⃣ Huấn luyện mô hình
```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
```

### 4️⃣ Đánh giá mô hình
```python
eval_result = model.evaluate(X_test, y_test)
print(f"Loss: {eval_result[0]}, Accuracy: {eval_result[1]}")
```

## 📊 Kết quả mong đợi
- Độ chính xác trên tập kiểm tra (**Test Accuracy**) khoảng **85-95%** tùy theo cách tiền xử lý dữ liệu.
- Có thể cải thiện bằng cách điều chỉnh tham số mô hình hoặc sử dụng kỹ thuật tăng cường dữ liệu (Data Augmentation).


