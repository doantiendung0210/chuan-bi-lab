# BÁO CÁO LÝ THUYẾT TUẦN 2: SUPERVISED & UNSUPERVISED LEARNING

---

## 1. Học có giám sát (Supervised Learning)

### 1.1. Khái niệm
**Supervised Learning (Học có giám sát)** là phương pháp học máy trong đó mô hình được huấn luyện trên một tập dữ liệu đã được gán nhãn (labeled data). Tập dữ liệu bao gồm các cặp đầu vào $X$ (features) và đầu ra mong muốn $y$ (label/target). 

Mục tiêu chính của mô hình là tìm ra một hàm ánh xạ $f: X \rightarrow y$ sao cho khi đưa vào một dữ liệu đầu vào mới chưa từng thấy, mô hình có thể dự đoán chính xác nhãn đầu ra tương ứng.

---

### 1.2. Quy trình tổng quát trong Supervised Learning

Một pipeline học máy có giám sát tiêu chuẩn gồm 7 bước chính:

1. **Thu thập và gán nhãn dữ liệu (Data Collection & Labeling):** Thu thập dữ liệu thô từ các nguồn uy tín và thực hiện gán nhãn chính xác cho dữ liệu đầu ra.
2. **Tiền xử lý dữ liệu (Data Preprocessing):**
   - Làm sạch dữ liệu (xử lý giá trị thiếu/missing values, loại bỏ outlier).
   - Chuẩn hóa dữ liệu (Feature Scaling: MinMaxScaler, StandardScaler).
   - Mã hóa dữ liệu dạng chuỗi/phân loại thành dạng số (Categorical Encoding: One-Hot Encoding, Label Encoding).
3. **Chia tập dữ liệu (Data Splitting):** Chia dữ liệu thành 3 tập riêng biệt:
- Tỷ lệ tham khảo:
   - **Train set (~70-80%):** Dùng để huấn luyện mô hình.
   - **Validation set (~10-15%):** Dùng để tinh chỉnh tham số (hyperparameters) và chọn mô hình.
   - **Test set (~10-15%):** Dùng để đánh giá độc lập hiệu năng cuối cùng của mô hình.
4. **Huấn luyện mô hình (Model Training):** Đưa dữ liệu tập Train vào các thuật toán để mô hình học các biểu diễn và trọng số thích hợp.
5. **Đánh giá mô hình (Model Evaluation):** Sử dụng tập Test/Validation với các thước đo (metrics) phù hợp:
   - Bài toán Classification: *Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC*.
   - Bài toán Regression: *MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE.
6. **Tối ưu hóa (Hyperparameter Tuning):** Tinh chỉnh các siêu tham số của mô hình (bằng *GridSearchCV*, *RandomizedSearchCV*, hoặc *Bayesian Optimization*) để đạt hiệu năng cao nhất và tránh hiện tượng Overfitting/Underfitting.
7. **Triển khai và giám sát (Deployment & Monitoring):** Tích hợp mô hình vào môi trường thực tế (API/Web) và liên tục giám sát độ chính xác theo thời gian thực.

---

### 1.3. Phân biệt Classification và Regression

| Tiêu chí | Classification (Phân loại) | Regression (Dự đoán giá trị) |
| :--- | :--- | :--- |
| **Bản chất biến đầu ra ($y$)** | Là giá trị rời rạc (Discrete / Categorical). | Là giá trị liên tục (Continuous numerical value). |
| **Mục tiêu bài toán** | Phân chia các điểm dữ liệu vào các nhóm/lớp cụ thể. | Dự đoán một chỉ số hoặc một mức độ định lượng cụ thể. |
| **Ví dụ thực tế** | Phân loại email (Spam / Non-spam), chẩn đoán bệnh (Bệnh / Không bệnh), nhận dạng chữ số viết tay. | Dự đoán giá nhà, dự đoán nhiệt độ thời tiết, dự đoán doanh thu bán hàng. |
| **Thước đo đánh giá tiêu biểu** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix. | MAE, MSE, RMSE, $R^2$ Score. |

---

## 2. Học không giám sát (Unsupervised Learning)

### 2.1. Khái niệm
**Unsupervised Learning (Học không giám sát)** là phương pháp học máy trong đó mô hình làm việc với tập dữ liệu đầu vào $X$ chưa được gán nhãn (unlabeled data). Mô hình tự khám phá các cấu trúc ẩn, mối quan hệ ngầm hoặc các quy luật tự nhiên bên trong dữ liệu mà không cần sự hướng dẫn của con người.

---

### 2.2. Mục tiêu và ứng dụng chính

- **Khám phá pattern ẩn (Hidden Pattern Discovery):** Tìm ra những quy luật phân bố hoặc mối tương quan không rõ ràng giữa các thuộc tính.
- **Gom nhóm dữ liệu (Clustering):** Tự động phân chia tập dữ liệu thành các cụm (cluster) sao cho các điểm dữ liệu trong cùng một cụm có độ tương đồng cao, còn các điểm thuộc khác cụm thì khác biệt rõ rệt.
  - *Ứng dụng:* Phân khúc khách hàng (Customer Segmentation), phân nhóm tài liệu/hình ảnh, phát hiện cấu trúc nhóm trong dữ liệu và hỗ trợ phát hiện bất thường.
- **Giảm chiều dữ liệu (Dimensionality Reduction) & Trực quan hóa:**
  - Nén tập dữ liệu có số lượng chiều/thuộc tính rất lớn về không gian ít chiều hơn (2D hoặc 3D) mà vẫn giữ được tối đa lượng thông tin quan trọng.
  - *Ứng dụng:* Trực quan hóa dữ liệu phức tạp, tăng tốc độ huấn luyện mô hình, khử nhiễu (Denoising).

---

### 2.3. Phân biệt Clustering và Dimensionality Reduction

| Tiêu chí | Clustering (Gom nhóm) | Dimensionality Reduction (Giảm chiều) |
| :--- | :--- | :--- |
| **Mục tiêu chính** | Nhóm các **mẫu dữ liệu (mẫu / dòng)** có đặc trưng giống nhau lại với nhau. | Tóm tắt/nén các **đặc trưng (features / cột)** của dữ liệu thành ít thuộc tính hơn. |
| **Đầu ra mong muốn** | Nhãn cụm (cluster IDs) đại diện cho từng phân nhóm dữ liệu. | Một không gian thuộc tính mới có số chiều thấp hơn ($k < d$). |
| **Thuật toán tiêu biểu** | K-Means, DBSCAN, Hierarchical Clustering. | PCA, t-SNE, UMAP. |

---

## 3. So sánh tổng quan Supervised vs Unsupervised Learning

| Tiêu chí | Supervised Learning (Học có giám sát) | Unsupervised Learning (Học không giám sát) |
| :--- | :--- | :--- |
| **Dữ liệu đầu vào** | Đã được gán nhãn ($X$ đi kèm nhãn $y$). | Chưa được gán nhãn (chỉ có $X$). |
| **Mục tiêu chính** | Dự đoán nhãn/giá trị đầu ra cho dữ liệu mới. | Khám phá cấu trúc ẩn, gom nhóm hoặc giảm chiều dữ liệu. |
| **Độ phức tạp & Chi phí** | Tốn nhiều chi phí và thời gian cho việc thu thập và gán nhãn dữ liệu ban đầu. | Dễ thu thập dữ liệu thô hơn, nhưng việc đánh giá kết quả mô hình phức tạp và mang tính chủ quan hơn. |
| **Đánh giá hiệu năng** | Dễ dàng đo lường bằng so sánh trực tiếp giá trị dự đoán với nhãn thực tế ($y$). | Khó đánh giá chính xác, thường dùng chỉ số Silhouette Score, Inertia hoặc trực quan hóa. |
| **Ứng dụng phổ biến** | Nhận diện khuôn mặt, phân loại thư rác, dự đoán chứng khoán. | Phân khúc thị trường, trực quan hóa dữ liệu gen, phát hiện bất thường. |

---

## 4. Giới thiệu các thuật toán tiêu biểu

### 4.1. Thuật toán Học có giám sát (Supervised Learning)
1. **Linear Regression (Hồi quy tuyến tính):** Tìm đường thẳng/siêu phẳng biểu diễn mối quan hệ tuyến tính giữa biến đầu vào và biến đầu ra liên tục.
2. **Logistic Regression (Hồi quy Logistic):** Thuật toán phân loại dựa trên hàm Sigmoid để tính xác suất thuộc về một lớp.
3. **Decision Tree (Cây quyết định):** Phân chia dữ liệu thành các nhánh dựa trên chuỗi quy tắc kiểm tra điều kiện (Decision Rules) thu được từ thuộc tính.
4. **Random Forest (Rừng ngẫu nhiên):** Mô hình Ensemble hợp nhất kết quả từ nhiều cây quyết định độc lập để tăng độ chính xác và giảm Overfitting.
5. **Support Vector Machine (SVM):** Tìm siêu phẳng (hyperplane) tối ưu phân tách các lớp dữ liệu với lề (margin) lớn nhất.
6. **K-Nearest Neighbors (KNN):** Phân loại hoặc dự đoán dữ liệu dựa trên nhãn của $K$ điểm dữ liệu lân cận gần nhất.

### 4.2. Thuật toán Học không giám sát (Unsupervised Learning)
1. **K-Means Clustering:** Gom nhóm dữ liệu dựa trên việc phân chia các điểm vào $K$ cụm sao cho tổng khoảng cách tới tâm cụm (centroid) là nhỏ nhất.
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Gom nhóm dựa trên mật độ dữ liệu, có khả năng phát hiện các cụm với hình dạng bất kỳ và xác định các điểm nhiễu (noise/outliers) mà không cần chỉ định trước số lượng cụm.
3. **Hierarchical Clustering (Phân cụm phân cấp):** Xây dựng cây phân cụm (Dendrogram) bằng cách gộp dần (Agglomerative) hoặc tách dần (Divisive) các tập hợp điểm.
4. **PCA (Principal Component Analysis - Phân tích thành phần chính):** Giảm chiều dữ liệu bằng cách chiếu dữ liệu lên các trục tọa độ mới (Principal Components) sao cho bảo toàn phương sai lớn nhất.
5. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Kỹ thuật giảm chiều phi tuyến tính rất hiệu quả để trực quan hóa dữ liệu nhiều chiều trên không gian 2D/3D.
