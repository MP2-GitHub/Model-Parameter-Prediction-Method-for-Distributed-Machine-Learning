import numpy as np
import gzip
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_SIZE = 32  # 目标图像大小
CHANELS_SIZE = 3  # RGB通道数
OUTPUT_SIZE = 10  # MNIST类别数


def preprocess_mnist_images(images):
    # 调整图像大小从28x28到32x32
    images_resized = tf.image.resize(images, [IMAGE_SIZE, IMAGE_SIZE])
    # 将灰度图像转换为RGB
    images_rgb = tf.image.grayscale_to_rgb(images_resized)
    return images_rgb


def load_dataset():
    def extract_images(filename):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)  # Skip header
            buf = bytestream.read()
            data = np.frombuffer(buf, dtype=np.uint8)
            num_images = data.shape[0] // (28 * 28)
            data = data.reshape(num_images, 28, 28, 1)
            return data

    def extract_labels(filename):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)  # Skip header
            buf = bytestream.read()
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    train_images = extract_images('./mnist_data/train-images-idx3-ubyte.gz')
    train_labels = extract_labels('./mnist_data/train-labels-idx1-ubyte.gz')
    test_images = extract_images('./mnist_data/t10k-images-idx3-ubyte.gz')
    test_labels = extract_labels('./mnist_data/t10k-labels-idx1-ubyte.gz')

    # 归一化图像到 [0, 1] 范围
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 预处理图像调整为32x32并转换为RGB
    with tf.Graph().as_default():
        images_tensor = tf.convert_to_tensor(train_images, dtype=tf.float32)
        preprocessed_train_images = preprocess_mnist_images(images_tensor)

        images_tensor = tf.convert_to_tensor(test_images, dtype=tf.float32)
        preprocessed_test_images = preprocess_mnist_images(images_tensor)

        with tf.Session() as sess:
            train_images = sess.run(preprocessed_train_images)
            test_images = sess.run(preprocessed_test_images)

    # 进行One-hot编码
    train_labels = np.eye(OUTPUT_SIZE)[train_labels]
    test_labels = np.eye(OUTPUT_SIZE)[test_labels]

    return train_images, train_labels, test_images, test_labels


def show_images(images, labels, num_images=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {np.argmax(labels[i])}")
        plt.axis('off')
    plt.show()


# 示例：加载数据集并打印一些信息
train_images, train_labels, test_images, test_labels = load_dataset()
print(f"训练图像形状: {train_images.shape}")
print(f"训练标签形状: {train_labels.shape}")
print(f"测试集图像形状: {test_images.shape}")
print(f"测试集标签形状: {test_labels.shape}")

# 显示前5张图像
show_images(train_images, train_labels, num_images=5)
