# Image-Classification-on-CIFAR-10_
This project evaluates multiple traditional machine learning classifiers and compares their performance on the CIFAR-10 image dataset. The focus is on assessing classical algorithms (like KNN, Decision Tree, Random Forest, Naive Bayes) applied to flattened image data.
# Implementation 
Load and preprocess CIFAR-10 image data

Flatten image data into feature vectors

Train and evaluate multiple classical ML classifiers:

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Naive Bayes

Compare performance using precision, recall, and F1-score
| Library        | Purpose                               |
| -------------- | ------------------------------------- |
| `NumPy`        | Matrix and numerical operations       |
| `TensorFlow`   | Loading CIFAR-10 dataset              |
| `scikit-learn` | Model training and evaluation metrics |

Dataset: CIFAR-10
Shape: 60,000 color images (32x32 pixels, 3 channels)

Classes: 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)

Split: 50,000 training and 10,000 test samples

# Methodology
Data Preprocessing

CIFAR-10 images are flattened from (32, 32, 3) to 1D vectors of size 3072

Normalization applied to pixel values (scaled to [0, 1])

Model Training

Trained 4 classifiers using scikit-learn:

DecisionTreeClassifier

RandomForestClassifier

GaussianNB

KNeighborsClassifier

Evaluation Metrics

Precision

Recall

F1 Score
# Outcomes 
| Model         | Precision | Recall | F1 Score |
| ------------- | --------- | ------ | -------- |
| Decision Tree | 0.28      | 0.27   | 0.27     |
| Random Forest | 0.41      | 0.39   | 0.38     |
| Naive Bayes   | 0.28      | 0.29   | 0.28     |
| KNN           | 0.38      | 0.36   | 0.36     |


