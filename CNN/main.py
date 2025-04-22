import tensorflow as tf
from MNISTData import MNISTData
from cnn import CNN

def main():
    # MNIST 데이터 로드
    mnist_data = MNISTData()
    mnist_data.load_data()  # 데이터를 로드하고 전처리 수행

    # 학습 및 테스트 데이터 가져오기
    train_images, train_labels = mnist_data.x_train, mnist_data.y_train
    test_images, test_labels = mnist_data.x_test, mnist_data.y_test

    # 데이터 차원 확장 (CNN 입력에 맞게 4D로 변환)
    train_images = train_images.reshape(-1, mnist_data.width, mnist_data.height, 1)
    test_images = test_images.reshape(-1, mnist_data.width, mnist_data.height, 1)

    # 레이블을 원-핫 인코딩
    train_labels = tf.keras.utils.to_categorical(train_labels, mnist_data.num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, mnist_data.num_classes)

    # CNN 모델 생성
    cnn = CNN(hidden_layer_conf=[32, 64], num_output_nodes=mnist_data.num_classes)
    cnn.image_shape_x = mnist_data.width  # 입력 이미지의 가로 크기
    cnn.image_shape_y = mnist_data.height  # 입력 이미지의 세로 크기
    cnn.num_labels = mnist_data.num_classes  # 클래스 수
    cnn.build_CNN_model()

    # 모델 학습
    cnn.classifier.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

    # 모델 평가
    test_loss, test_acc = cnn.classifier.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

if __name__ == "__main__":
    main()