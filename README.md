# ECEN 758 Project - Data Mining and Analysis (Fall 2025)

## Team: Group 6

### Team Members
- Harine Choi
- Wei-Chen Huang
- Harshavardhan Reddy Varicuti
- Shou-En Liu
### Abstract
This project examines several approaches for classifying news articles in the Sogou News dataset, which contains about 510,000 samples across five categories. We built a complete pipeline that includes dataset download from Hugging Face, data cleaning and normalization, exploratory analysis, and multiple supervised text classification models. Our main baselines use character-level TF-IDF features combined with three standard classifiers: Linear Support Vector Machines (LinearSVC), Multinomial Naive Bayes (MNB), and k-Nearest Neighbors (KNN). Trained on 450,000 examples and evaluated on the official 60,000-sample test split, the TF-IDF + LinearSVC model reached roughly 90% macro-F1 and 90% accuracy, outperforming the other classical baselines.
We further explored word-level TF-IDF with LinearSVC using randomized search, grid search, and Bayesian hyperparameter optimization to tune n-gram and regularization settings. To compare these feature-engineering methods with representation learning, we also implemented two convolutional neural networks: a basic 1D CNN and a TextCNN model with multiple kernel sizes and batch normalization. Both neural models operate on tokenized title-and-content sequences and are trained end-to-end with learned embeddings. The overall framework provides a clear comparison between sparse TF-IDF models and CNN-based models under a shared preprocessing pipeline, and it serves as a solid baseline for future work with deeper neural networks or transformer-based encoders.

### Dataset
- Name: SogouNews
- Number of Classes: Multi-class (more than 5)
- Brief Description: SogouNews is a Chinese news dataset containing articles categorized into multiple classes. Our task is to classify the news into their corresponding categories.

### Model
- Model type: Multinomial Naive Bayes (MNB)/k-Nearest Neighbors (KNN)/Linear Support Vector Classifier (LinearSVC)/1D CNN/TextCNN
- Short Description: We train our model to classify news articles from SogouNews dataset into multiple categories. The model uses [briefly describe your preprocessing, embeddings, or architecture]. We evaluate performance using standard metrics such as accuracy and F1-score.

### Results
1. Evaluation Metrics
LinearSVC + TF-IDF (1-3 n-gram)
Validation Accuracy: ~97%
Test Macro-F1: ~90%
Other Models:
KNN: ~96% accuracy
Naive Bayes: ~91% accuracy
1D CNN: ~95% accuracy, ~88% macro-F1
TextCNN: ~96% accuracy, ~89% macro-F1

2. Confusion Matrix
LinearSVC Confusion Matrix:
Analysis:
Most classes were classified correctly with high precision and recall.
The main confusion occurred between Entertainment and Finance, likely due to overlapping vocabulary in pinyin.
Other categories, such as Sports and Technology, had very few cross-class errors.
<img src="Confusion Matrix.png" width="60%">
0-sports, 1-finance, 2-entertainment, 3-automobile, 4-technology
<img src="class_distribution.png" width="60%">
<img src="dim_reduction.png" width="60%">
<img src="text_length_statistics.png" width="60%">

### Code & Demo
- GitHub code: [GitHub Notebook]https://github.com/wellsonhuang/ECEN-758-Fall-2025-Project

### Notes
- This website summarizes the project completed for ECEN 758, focusing on multi-class class
