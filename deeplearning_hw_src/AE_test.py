# 모델 학습 결과를 사용하여 과제 수행
import numpy as np
import matplotlib.pyplot as plt
from AutoEncoder import AutoEncoder
from MNISTData import MNISTData
from AE_train import noise_adder

# data loading
data = MNISTData()
data.load_data()

# Load the model
model = AutoEncoder()
model.input_output_dim = data.in_out_dim
model.build_model()
model.load_weights("model/autoEncoder.weights.h5")

# Create noise test data
x_test_noised = noise_adder(x=data.x_test, drop_prob=0.5)

# Reconstructed
reconstructed = model.en_decoder.predict(x_test_noised)

# [2번] Denoising 성능 확인 ============================

# Visualization
def visualize_56_pairs(noised_images, reconstructed_images, labels):
    num_row = 7
    num_col = 8
    plt.figure(figsize=(14, 10))
    plt.suptitle("Digit pairs", fontsize=16)

    for i in range(num_row * num_col):
        # Noised Image
        plt.subplot(num_row, num_col * 2, 2 * i + 1)
        plt.imshow(noised_images[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.text(14, 31, str(labels[i]), fontsize=8, ha='center', va='top')

        # Reconstructed Image
        plt.subplot(num_row, num_col * 2, 2 * i + 2)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.text(14, 31, str(labels[i]), fontsize=8, ha='center', va='top')

    plt.subplots_adjust(top=0.92)
    plt.show()

num_visual = 56
visual_noised = x_test_noised[:num_visual]
visual_recon = reconstructed[:num_visual]
visual_labels = data.y_test[:num_visual]

visualize_56_pairs(visual_noised, visual_recon, visual_labels) # [2번] 실행행

# [3번] 클래스별 code 평균값으로 이미지 생성 ===============================
 
# data loading and encoding 
def load_and_encode_data(model, data):
    x_input = data.x_test[:1000]
    y_input = data.y_test[:1000]
    codes = model.encoder.predict(x_input) # shape : (1000, code_dim)
    return x_input, y_input, codes

# 클래스별 평균 code 계산 메서드
def compute_class_avg_codes(codes, labels):
    class_avg_codes = []
    for digit in range(10):
        class_codes = codes[labels == digit]
        avg_code = np.mean(class_codes, axis=0)
        class_avg_codes.append(avg_code)
    return np.array(class_avg_codes) # shape : (10, code_dim)

x_input, y_input, codes = load_and_encode_data(model, data)
class_avg_codes = compute_class_avg_codes(codes, y_input)

# 평균 code -> Decoder -> 이미지 생성
gen_images = model.decoder.predict(class_avg_codes) # shape : (10, 784)
gen_images = np.clip(gen_images, 0.0, 1.0) # 0~1 사이로 clip

# visualization
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(gen_images[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.text(14, 31, str(i), fontsize=8, ha='center', va='top')
plt.suptitle("Generated images from mean code per digit", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()

# ========================================================================

# [4번] 클래스별 code 평균값 및 표준편차를 활용하여 숫자 생성 ================

x_input, y_input, codes = load_and_encode_data(model, data)

# class별 평균 및 표준편차 code 계산
def compute_class_avg_std_codes(codes, labels):
    class_avg_codes = []
    class_std_codes = []
    for digit in range(10):
        class_codes = codes[labels == digit]
        avg_code = np.mean(class_codes, axis=0)
        std_code = np.std(class_codes, axis=0)
        class_avg_codes.append(avg_code)
        class_std_codes.append(std_code)
    return np.array(class_avg_codes), np.array(class_std_codes) # shape : (10, code_dim)

class_avg_codes, class_std_codes = compute_class_avg_std_codes(codes, y_input)

# 랜덤 벡터를 통한 새 code 생성 및 재구성
num_variations = 5
gen_codes = []

for i in range(10):
    for _ in range(num_variations):
        rand_j = np.random.uniform(-1, 1, size=class_avg_codes[i].shape)
        new_code = class_avg_codes[i] + class_std_codes[i] * rand_j
        gen_codes.append(new_code)

gen_codes = np.array(gen_codes) # shape : (50, code_dim)

# decoder로 이미지 생성 
gen_images = model.decoder.predict(gen_codes)
gen_images = np.clip(gen_images, 0.0, 1.0) # 0~1 사이로 clip

# visualization
plt.figure(figsize=(10, 10))
for i in range(10):
    for j in range(num_variations):
        idx = i * num_variations + j
        plt.subplot(10, 5, idx + 1)
        plt.imshow(gen_images[idx].reshape(28, 28), cmap="gray")
        plt.axis("off")
plt.suptitle("Generated images from mean & std codes", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
# ========================================================================

# [5번] 클래스별 code 평균값 위치관계 확인 ==================================

x_input, y_input, codes = load_and_encode_data(model, data)

# class별 평균 및 표준편차 code 계산
class_avg_codes, class_std_codes = compute_class_avg_std_codes(codes, y_input)

# 평균 제거 (정규화)
mean_centered = class_avg_codes - np.mean(class_avg_codes, axis=0)

# 공분산 행렬 계산 
cov_matrix = np.cov(mean_centered.T)


# 고유값 분해
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# 고유값 기준 정렬 (내림차순)
sorted_indices = np.argsort(-eig_vals)
top2_eig_vecs = eig_vecs[:, sorted_indices[:2]]

# 2차원 축소된 좌표 계산 
reduced = mean_centered @ top2_eig_vecs # shape (10, 2)

# t-SNE와 유사한 방식으로 노이즈 추가
np.random.seed(42)
noise = np.random.normal(scale=0.5, size=reduced.shape)
reduced_tsne_like = reduced + noise

# visualization
plt.figure(figsize=(8, 6))
for i in range(10):
    x, y = reduced[i]
    plt.scatter(x, y, s=100, color='darkcyan')
    plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white', fontweight='bold')

plt.title("t-SNE Visualization of Avg Codes (per digit)", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.show()
# # ========================================================================