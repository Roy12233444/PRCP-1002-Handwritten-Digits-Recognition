# Handwritten Digit Recognition System

A comprehensive machine learning project that implements various approaches for handwritten digit recognition, including classical machine learning models, deep learning architectures, and attention mechanisms.

## 📋 Project Overview

This project explores multiple machine learning and deep learning techniques for recognizing handwritten digits. It includes implementations of classical ML models, CNNs, and advanced attention mechanisms, along with comprehensive data analysis and model evaluation.

## 🚀 Features

- **Classical ML Models**:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Random Forest
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - Decision Trees

- **Deep Learning Models**:
  - Basic Neural Networks
  - Convolutional Neural Networks (CNN)
  - Recurrent Convolutional Neural Networks (CRNN)
  - Attention Mechanisms
  - Vision Transformers

- **Advanced Features**:
  - Data Augmentation Pipeline
  - Feature Engineering
  - Model Evaluation and Comparison
  - Multi-digit Recognition
  - Model Deployment

## 📂 Project Structure

```
.
├── ATTENTION MECHANISM_MODELS/    # Models with attention mechanisms
├── AUGMENTATION MODEL/           # Data augmentation pipeline
├── Basic_Neural_Network_Models/  # Basic neural network implementations
├── CNN_MODEL_CSV_FILE/           # CNN models and training history
├── CRNN_MODEL_MULTI_DIGIT_RECOGNITION/  # Multi-digit recognition models
├── Classical_ML_Models/          # Classical ML model implementations
├── Datasets_of_Feature_Engineering/  # Processed datasets
├── PRCP-1002-Handwritten_Digits_Recognition_Final_File3ipynb.json  # Main notebook
├── Complete_Data_Analysis_Report.md  # Detailed analysis report
└── Challenges_and_Solutions_Report.md  # Project challenges and solutions
```

## 🛠️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Roy12233444/PRCP-1002-Handwritten-Digits-Recognition.git
   cd PRCP-1002-Handwritten-Digits-Recognition
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: If requirements.txt is not provided, install the following packages:
   ```
   numpy pandas matplotlib seaborn scikit-learn tensorflow torch torchvision opencv-python xgboost
   ```

## 🚦 Usage

1. **Running the Jupyter Notebook**:
   ```bash
   jupyter notebook PRCP-1002-Handwritten_Digits_Recognition_Final_File3ipynb.json
   ```

2. **Training Models**:
   - The notebook contains separate sections for different model types
   - Follow the cells in sequence to train and evaluate models

3. **Using Pre-trained Models**:
   - Load any pre-trained model from the respective model directories
   - Example for loading a CNN model:
     ```python
     import tensorflow as tf
     model = tf.keras.models.load_model('CNN_MODEL_CSV_FILE/advanced_cnn_model.h5')
     ```

## 📊 Results

Detailed performance metrics and comparisons between different models are available in the notebook and the `Complete_Data_Analysis_Report.md`.

## 📝 Notes

- The project uses Git LFS for large files. Make sure you have Git LFS installed:
  ```bash
  git lfs install
  ```
- Some model files are large and may take time to download

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any queries, please open an issue or contact the repository owner.