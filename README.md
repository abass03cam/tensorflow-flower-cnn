# 🌸 Flower Classification with TensorFlow

A deep learning project using TensorFlow and Keras to classify flower images into five categories with a Convolutional Neural Network (CNN). The model is trained on the publicly available `flower_photos` dataset and can predict the type of flower in new images.

---

## 📌 Features

- CNN with 3 convolutional layers
- Image preprocessing and data augmentation
- Training/Validation data split
- Predict new images with class confidence
- 5 flower classes: daisy, dandelion, roses, sunflowers, tulips

---

## 🧠 Model Architecture

- Input: 180x180 RGB images
- Convolutional layers: 16, 32, 64 filters
- Dense layers: 128 units + output layer with softmax
- Optimizer: Adam
- Loss: sparse categorical crossentropy

---

## 📂 Dataset

Dataset used: [TensorFlow flower_photos](https://www.tensorflow.org/datasets/catalog/tf_flowers)

Download manually or use the following path structure:
