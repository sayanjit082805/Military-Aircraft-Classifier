# Military-Aircraft-Classifier

A deep learning project to classify military aircraft into nearly 81 distinct aircraft types. The model is trained on the Military Aircraft Detection dataset from Kaggle with PASCAL VOC format annotations and uses a ResNet (Residual Neural Network) architecture for classification.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Overview

A deep learning project to classify military aircraft into nearly 81 distinct aircraft types. The model is trained on the [Military Aircraft Detection dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data) from Kaggle with PASCAL VOC format annotations and uses a ResNet architecture for classification.

# Dataset

The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data). It contains around 36,000 images of aircraft of various different countries.

**Dataset Statistics** :

- **Total Images**: ~36,000
- **Training Set**: ~25,000 images
- **Test Set**: ~4,000 images
- **Validation Set**: ~7,000 images
- **Classes**: ~81 distinct military aircraft types

# Installation

I strongly recommend against installing locally due to the large dataset size. Instead, you can check out this [Kaggle notebook](https://www.kaggle.com/code/sayanjit082805/notebook5102672fcb).

Alternatively, just visit the site.

# Model Architecture

The model uses the ResNet-50 architecture. Additional fine-tuning has been performed, to increase the accuracy. Global Average Pooling has been used to reduce overfitting and the optimiser used is the adam optimiser.

> [!NOTE]
> The model is not intended for real-world applications and should not be used for any commercial or operational purposes.

# Acknowledgments

- The dataset has been taken from Kaggle.
- All images used are sourced from publicly available search engines (Google Images, DuckDuckGo).

# License

This project is licensed under The Unlicense License, see the LICENSE file for details.

> [!NOTE]
> The License does not cover the dataset. It has been taken from Kaggle.
