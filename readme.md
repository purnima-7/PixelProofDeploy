# PixelProof

PixelProof is a lightweight classical computer vision system that distinguishes between a **real-world photograph** and a **photograph of a screen**.

Unlike deep learning approaches, this project relies entirely on handcrafted image-processing features and traditional machine learning, making it lightweight, interpretable, and suitable for CPU-only inference.

---

## Features

The system extracts a variety of handcrafted image features including:

- Laplacian Variance (blur/sharpness)
- Edge Density
- Local Binary Patterns (LBP)
- Gray Level Co-occurrence Matrix (GLCM)
- FFT Peak Analysis
- RGB Color Statistics
- HSV Color Statistics
- Brightness
- Contrast
- Shannon Entropy
- Glare Percentage
- Pixel Grid Score
- Hough Line Features

These features are used to train classical machine learning classifiers.

---

## Models Evaluated

The following models were compared using 5-fold Stratified Cross Validation:

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The model with the highest validation accuracy is automatically selected and saved.

---

## Dataset Structure

```
dataset/
│
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── screen/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## Project Structure

```
PixelProof/
│
├── dataset/
│   ├── real/
│   └── screen/
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│   ├── error_analysis.py
│   └── features.csv
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd PixelProof
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Generate Features

Run

```bash
cd src

python feature_extraction.py
```

This extracts handcrafted features from every image and creates

```
features.csv
```

---

## Train the Model

Run

```bash
python train.py
```

The script

- Performs 5-fold Stratified Cross Validation
- Compares Random Forest, SVM and XGBoost
- Reports Accuracy, Precision, Recall, F1-score and ROC-AUC
- Saves the best model as

```
models/best_model.pkl
```

---

## Error Analysis

To inspect misclassified images, run

```bash
python error_analysis.py
```

This generates:

- Confusion Matrix
- Classification Report
- List of misclassified images

---

## Predict a New Image

Run

```bash
python predict.py path/to/image.jpg
```

Example

```bash
python predict.py dataset/real/sample.jpg
```

Output

```
Score      : 0.0021
```

or

```
Score      : 0.9984
```

The score represents the probability that the input image is a photograph of a screen.

- Score close to **0** → Real Photo
- Score close to **1** → Screen Photo

---

## Performance

Using 5-fold Stratified Cross Validation:

| Metric | Score |
|---------|-------|
| Accuracy | ~96% |
| Precision | ~96% |
| Recall | ~96% |
| F1 Score | ~96% |
| ROC-AUC | ~0.99 |

---

## Inference Time

Measured on a Windows laptop CPU.

| Stage | Average Time |
|--------|--------------|
| Feature Extraction | ~250–300 ms |
| Prediction | ~60–90 ms |
| Total | ~320–390 ms |

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- scikit-image
- Pillow

---

## Future Improvements

Possible improvements include:

- Faster FFT implementation
- Parallel feature extraction
- Additional frequency-domain descriptors
- Mobile deployment
- Hybrid classical + deep learning approach
- Larger and more diverse training dataset

---

## License

This project was developed as part of an image classification assignment for educational purposes.