# BÁO CÁO TÌM HIỂU LÝ THUYẾT MACHINE LEARNING

## 1. Ensemble Learning

**Ensemble Learning (học tổ hợp)** là phương pháp kết hợp dự đoán từ nhiều mô hình để tạo ra kết quả cuối cùng. Thay vì phụ thuộc vào một mô hình, phương pháp này tận dụng điểm mạnh của từng mô hình.

**Tại sao cần kết hợp nhiều mô hình?**
- **Tăng độ chính xác:** nhiều mô hình có thể bổ sung cho nhau.
- **Giảm overfitting:** hạn chế việc mô hình học quá sát dữ liệu huấn luyện, đặc biệt với Bagging.
- **Tăng tính ổn định:** giảm tác động của sai số từ một mô hình riêng lẻ.

### Các kỹ thuật phổ biến

| Kỹ thuật | Nguyên lý | Đặc điểm / ví dụ |
|---|---|---|
| **Bagging** | Huấn luyện nhiều mô hình độc lập trên các mẫu dữ liệu được lấy ngẫu nhiên có hoàn lại, sau đó tổng hợp kết quả. | Giảm phương sai; **Random Forest** kết hợp nhiều cây quyết định và chọn ngẫu nhiên đặc trưng khi chia nhánh. |
| **Boosting** | Huấn luyện các mô hình theo trình tự, mỗi mô hình mới tập trung cải thiện lỗi của tổ hợp trước đó. | **AdaBoost** tăng trọng số cho các mẫu bị dự đoán sai; **XGBoost** dùng gradient boosting tối ưu hóa hàm mất mát và bổ sung cơ chế regularization. |
| **Stacking** | Huấn luyện nhiều mô hình cơ sở, sau đó dùng một mô hình khác (*meta-model*) học cách kết hợp dự đoán của chúng. | Có thể phối hợp Logistic Regression, SVM, Random Forest; nên dùng dự đoán out-of-fold để tránh rò rỉ dữ liệu. |
| **Voting Classifier** | Kết hợp kết quả từ nhiều mô hình phân loại. | **Hard voting:** chọn nhãn được bỏ phiếu nhiều nhất. **Soft voting:** lấy trung bình xác suất dự đoán (có thể có trọng số). |

**Ví dụ:** Để phân loại email spam, có thể kết hợp Logistic Regression, Decision Tree và Naive Bayes bằng Voting Classifier thay vì chỉ sử dụng một mô hình.

## 2. Semi-Supervised Learning

**Semi-Supervised Learning (học bán giám sát)** sử dụng đồng thời **một lượng nhỏ dữ liệu có nhãn** và **một lượng lớn dữ liệu chưa có nhãn** để huấn luyện mô hình.

**Khi nào hữu ích?** Khi việc gán nhãn tốn nhiều thời gian hoặc chi phí, chẳng hạn ảnh y tế cần bác sĩ đánh giá, dữ liệu âm thanh cần phiên âm hoặc tài liệu cần phân loại thủ công.

### Các phương pháp cơ bản

- **Pseudo-Labeling:** huấn luyện trên dữ liệu có nhãn, dùng mô hình dự đoán nhãn cho dữ liệu chưa gán nhãn, rồi bổ sung các dự đoán đủ tin cậy vào tập huấn luyện.
- **Self-Training:** lặp lại quá trình tự gán nhãn và huấn luyện để mở rộng dữ liệu học.
- **Label Propagation:** lan truyền nhãn giữa các điểm dữ liệu có độ tương đồng cao, thường dựa trên đồ thị.
- **Consistency Regularization:** khuyến khích mô hình đưa ra dự đoán nhất quán khi đầu vào bị biến đổi nhẹ (ví dụ xoay hoặc thay đổi độ sáng ảnh).

**Lưu ý:** Nhãn tự sinh có thể sai và khiến lỗi lan rộng. Cần chọn ngưỡng tin cậy, đánh giá bằng tập dữ liệu có nhãn riêng và kiểm soát chất lượng dữ liệu.

## 3. Probabilistic Graphical Models (PGM)

**Probabilistic Graphical Models (mô hình đồ thị xác suất)** biểu diễn các biến ngẫu nhiên và quan hệ phụ thuộc giữa chúng bằng **đồ thị**, giúp mô hình hóa sự không chắc chắn và suy luận xác suất.

### Hai loại mô hình chính

| Tiêu chí | Bayesian Networks | Markov Networks |
|---|---|---|
| Cấu trúc | Đồ thị **có hướng, không chu trình** (DAG). | Đồ thị **vô hướng**. |
| Biểu diễn | Các cạnh thể hiện quan hệ phụ thuộc có điều kiện; phân phối chung được phân tích thành tích các xác suất có điều kiện. | Các cạnh thể hiện tương tác giữa các biến; phân phối chung được biểu diễn bằng các hàm thế (*potential functions*) và hệ số chuẩn hóa. |
| Ví dụ | Mô hình quan hệ giữa bệnh, triệu chứng và kết quả xét nghiệm. | Mô hình tương tác giữa các pixel lân cận khi phân đoạn ảnh. |

### Ứng dụng trong Machine Learning

- **Chẩn đoán và hỗ trợ quyết định:** suy ra xác suất bệnh từ các triệu chứng và bằng chứng quan sát được.
- **Xử lý ngôn ngữ tự nhiên:** mô hình hóa quan hệ giữa các nhãn hoặc từ trong chuỗi.
- **Thị giác máy tính:** phân đoạn ảnh, khử nhiễu và mô hình hóa tương tác không gian.
- **Phát hiện bất thường:** đánh giá xác suất của các trạng thái hoặc chuỗi sự kiện.

**Ưu điểm:** biểu diễn được sự không chắc chắn và các quan hệ giữa biến. **Hạn chế:** việc học và suy luận có thể tốn nhiều tài nguyên khi đồ thị phức tạp.

## 4. Recommendation Systems

**Recommendation Systems (hệ thống gợi ý)** dự đoán nội dung hoặc sản phẩm mà người dùng có khả năng quan tâm, dựa trên sở thích, hành vi và thông tin về đối tượng được gợi ý.

### Các phương pháp cơ bản

| Phương pháp | Cách hoạt động | Ưu điểm | Hạn chế |
|---|---|---|---|
| **Content-Based Filtering** | Gợi ý các mục có đặc điểm tương tự những mục người dùng từng thích (thể loại, tác giả, từ khóa...). | Không cần dữ liệu từ nhiều người dùng khác. | Dễ gợi ý lặp lại, ít khám phá sở thích mới. |
| **Collaborative Filtering** | Dựa vào tương tác của nhiều người dùng để tìm người có sở thích tương tự hoặc các mục thường được quan tâm cùng nhau. | Khám phá được sở thích không thể hiện trực tiếp trong mô tả nội dung. | Khó với người dùng/sản phẩm mới (*cold start*); dữ liệu tương tác thường thưa. |

**Ví dụ ứng dụng:**
- **Netflix:** gợi ý phim/chương trình dựa trên lịch sử xem, sở thích và tín hiệu về nội dung.
- **Amazon:** đề xuất sản phẩm liên quan đến hành vi xem, mua hàng và mối liên hệ giữa sản phẩm.
- **Spotify:** gợi ý bài hát, nghệ sĩ và playlist dựa trên hành vi nghe cùng đặc trưng âm nhạc.

Trong thực tế, các nền tảng lớn thường sử dụng **hệ thống lai (Hybrid Recommendation)**, kết hợp nhiều nguồn tín hiệu và nhiều mô hình, thay vì chỉ một phương pháp duy nhất.

## 5. Giới thiệu MLOps

**MLOps (Machine Learning Operations)** là tập hợp quy trình và công cụ để **phát triển, triển khai, theo dõi và duy trì mô hình ML** một cách có hệ thống trong môi trường thực tế.

**Vì sao cần MLOps?** Một mô hình có kết quả tốt trong thử nghiệm chưa chắc hoạt động tốt sau khi triển khai. Dữ liệu thực tế có thể thay đổi, mô hình có thể suy giảm chất lượng, và việc cập nhật thủ công dễ gây lỗi. MLOps giúp tự động hóa, quản lý phiên bản, tái lập thí nghiệm và giám sát mô hình.

### ML Pipeline là gì?

**ML Pipeline** là chuỗi bước xử lý dữ liệu và mô hình từ đầu vào đến hệ thống sử dụng thực tế. Một pipeline điển hình:

**Thu thập dữ liệu → Làm sạch / Tiền xử lý → Chia tập dữ liệu → Huấn luyện → Đánh giá → Triển khai → Giám sát → Huấn luyện lại**

### Bốn giai đoạn vận hành chính

1. **Model Training:** huấn luyện mô hình bằng dữ liệu đã xử lý; thử nghiệm siêu tham số và đánh giá trên tập dữ liệu phù hợp.
2. **Deployment:** đưa mô hình vào ứng dụng thông qua API, dịch vụ trực tuyến hoặc xử lý theo lô (*batch*).
3. **Monitoring:** theo dõi độ trễ, lỗi hệ thống, chất lượng dự đoán và thay đổi phân phối dữ liệu (*data drift*).
4. **Retraining:** cập nhật mô hình bằng dữ liệu mới khi hiệu quả giảm hoặc theo lịch; kiểm thử trước khi đưa phiên bản mới vào sử dụng.

### Một số công cụ phổ biến

| Công cụ | Công dụng chính |
|---|---|
| **MLflow** | Theo dõi thí nghiệm, lưu thông số, kết quả và quản lý phiên bản mô hình. |
| **DVC** | Quản lý phiên bản dữ liệu, mô hình và các bước pipeline. |
| **Docker** | Đóng gói ứng dụng và môi trường chạy mô hình để triển khai nhất quán. |
| **Kubeflow** | Xây dựng, điều phối các quy trình ML trên Kubernetes. |
| **Apache Airflow** | Lập lịch và điều phối các tác vụ trong pipeline. |
| **Evidently** | Theo dõi chất lượng dữ liệu, data drift và hiệu suất mô hình. |

**Ví dụ:** Một hệ thống dự đoán khách hàng có nguy cơ rời bỏ dịch vụ được huấn luyện từ dữ liệu lịch sử, triển khai qua API, theo dõi kết quả hằng tuần và huấn luyện lại khi hành vi khách hàng thay đổi.

## 6. Kết quả thực nghiệm và nhận xét

### 6.1. So sánh các mô hình Collaborative Filtering

| Chỉ số đánh giá | SVD | KNN |
|---|---|---|
| RMSE | **0.8807** | 0.9800 |
| MAE | **0.6766** | 0.7615 |
| Precision | **0.8095** | 0.7576 |
| Recall | **0.3287** | 0.2427 |

**Nhận xét:**

- **RMSE và MAE:** SVD có sai số dự đoán thấp hơn KNN, cho thấy khả năng dự đoán rating chính xác hơn.
- **Precision:** SVD đạt 0.8095, cao hơn KNN (0.7576), cho thấy tỷ lệ dự đoán tích cực chính xác cao hơn.
- **Recall:** SVD đạt 0.3287, cao hơn KNN (0.2427). Tuy nhiên, Recall của cả hai mô hình còn khá thấp, cho thấy vẫn bỏ sót nhiều lượt đánh giá tích cực.
- **Kết luận:** SVD cho kết quả tốt hơn KNN trên cả bốn chỉ số đánh giá trong thực nghiệm này.

### 6.2. So sánh với Content-Based Filtering (TF-IDF + Cosine Similarity)

| Tiêu chí | Collaborative Filtering (SVD) | Content-Based Filtering (TF-IDF) |
|---|---|---|
| Dữ liệu sử dụng | Lịch sử đánh giá của người dùng | Thể loại phim (genres) |
| Phương pháp | SVD (Matrix Factorization) | TF-IDF và Cosine Similarity |
| Nguyên lý | Dự đoán rating dựa trên sở thích tiềm ẩn của người dùng | Tìm các phim có thể loại tương đồng |
| Cá nhân hóa | Cao, dựa trên lịch sử rating | Chưa cá nhân hóa theo người dùng trong cách triển khai hiện tại |
| Kết quả thực nghiệm | RMSE = 0.8807; MAE = 0.6766 | Đã tính ma trận Cosine Similarity |
| Ưu điểm | Khả năng cá nhân hóa tốt | Đơn giản, dễ giải thích, không cần rating |
| Hạn chế | Phụ thuộc vào dữ liệu rating | Phụ thuộc vào thông tin thể loại phim |

**Nhận xét:**

- **Collaborative Filtering (SVD):** Khai thác lịch sử đánh giá của người dùng để dự đoán mức độ yêu thích đối với các bộ phim. Mô hình cho kết quả dự đoán tương đối tốt với RMSE = 0.8807 và MAE = 0.6766.
- **Content-Based Filtering (TF-IDF):** Sử dụng thông tin thể loại phim để tính toán độ tương đồng, từ đó xác định những bộ phim có nội dung tương tự. Phương pháp này không cần lịch sử đánh giá nhưng có thể hạn chế tính đa dạng của gợi ý.
- **So sánh thực nghiệm:** SVD được đánh giá bằng RMSE, MAE, Precision và Recall, trong khi TF-IDF mới được sử dụng để tính độ tương đồng. Do hai phương pháp chưa được đánh giá trên cùng một tiêu chí nên chưa thể kết luận phương pháp nào tốt hơn về chất lượng gợi ý.

### 6.3. Kết luận chung

Qua quá trình thực nghiệm, mô hình SVD cho kết quả tốt hơn KNN trong bài toán dự đoán rating, với sai số thấp hơn và Precision, Recall cao hơn.

Đối với Content-Based Filtering, TF-IDF kết hợp Cosine Similarity cho phép xác định mức độ tương đồng giữa các bộ phim dựa trên thể loại mà không cần dữ liệu đánh giá.

Nhìn chung, Collaborative Filtering phù hợp với bài toán cá nhân hóa đề xuất phim dựa trên lịch sử người dùng, trong khi Content-Based Filtering phù hợp với việc tìm kiếm các bộ phim tương tự dựa trên đặc trưng nội dung.

Hai phương pháp có thể bổ sung cho nhau trong một hệ thống gợi ý kết hợp (Hybrid Recommendation System).


## 7. Tìm hiểu về MLflow

### 7.1. MLflow là gì?
MLflow là công cụ mã nguồn mở giúp quản lý và theo dõi quá trình phát triển mô hình Machine Learning.

MLflow thuộc lĩnh vực MLOps, giúp việc huấn luyện, đánh giá và quản lý mô hình thuận tiện hơn.

### 7.2. Chức năng chính
- **Experiment Tracking:** Lưu các lần huấn luyện và kết quả đánh giá.
- **Parameters:** Lưu siêu tham số của mô hình như `max_depth`, `n_estimators`.
- **Metrics:** Lưu các chỉ số đánh giá như Accuracy, Precision, Recall.
- **Model Registry:** Quản lý các phiên bản mô hình.
- **MLflow UI:** Giao diện xem và so sánh các lần huấn luyện.

### 7.3. Ứng dụng trong bài thực hành
Sử dụng MLflow để theo dõi mô hình XGBoost trên bộ dữ liệu Titanic.

Các bước thực hiện:
1. Cài đặt thư viện `mlflow`.
2. Tạo experiment bằng `mlflow.set_experiment()`.
3. Bắt đầu tracking bằng `mlflow.start_run()`.
4. Huấn luyện mô hình XGBoost.
5. Lưu tham số bằng `mlflow.log_params()`.
6. Lưu Accuracy bằng `mlflow.log_metric()`.
7. Xem kết quả bằng MLflow UI.

**Kết quả:** Mô hình đạt Accuracy khoảng 81.56%.

### 7.4. Kết luận
MLflow giúp lưu lại tham số và kết quả huấn luyện, từ đó dễ dàng theo dõi, so sánh và cải thiện mô hình.



