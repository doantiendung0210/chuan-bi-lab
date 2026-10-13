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

## 6. Kết luận

Các chủ đề trên giải quyết những vấn đề khác nhau trong Machine Learning: **Ensemble Learning** nâng cao chất lượng dự đoán; **Semi-Supervised Learning** tận dụng dữ liệu thiếu nhãn; **PGM** mô hình hóa quan hệ xác suất; **Recommendation Systems** cá nhân hóa trải nghiệm; và **MLOps** giúp vận hành mô hình ổn định, có thể cập nhật trong thực tế.

## Tài liệu tham khảo

1. Scikit-learn User Guide – Ensemble Methods: https://scikit-learn.org/stable/modules/ensemble.html
2. Scikit-learn User Guide – Semi-supervised Learning: https://scikit-learn.org/stable/modules/semi_supervised.html
3. Koller, D. & Friedman, N. (2009), *Probabilistic Graphical Models: Principles and Techniques*, MIT Press.
4. Google Developers – Recommendation Systems: https://developers.google.com/machine-learning/recommendation
5. Google Cloud – MLOps: Continuous Delivery and Automation Pipelines in Machine Learning: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
6. MLflow Documentation: https://mlflow.org/docs/latest/index.html
