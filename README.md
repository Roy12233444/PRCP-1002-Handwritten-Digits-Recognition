# Handwritten Digit Recognition System

A production-oriented research project that benchmarks classical machine learning, deep learning, and attention-driven architectures on the MNIST handwritten digits corpus while delivering deployable artifacts, analytical insights, and optimization recipes for multiple inference targets.

## 📋 Overview

- **Scope**: End-to-end digit recognition covering raw data profiling, feature engineering, supervised learning, transfer learning, and deployment-ready compression.
- **Deliverables**: Executable notebook, curated model zoo, enriched feature datasets, augmentation pipelines, comparative reports, and optimization playbooks.
- **Objective**: Provide a reusable reference stack that balances accuracy, latency, and footprint for scenarios ranging from academic experimentation to edge deployment.

## 🚀 Highlights

- **11+ feature families** stitched into comprehensive 1,511-dimensional descriptors alongside PCA and LDA reductions for classical models.
- **Model portfolio** spanning Logistic Regression, SVM, Random Forest, XGBoost, KNN, Decision Trees, dense neural nets, CNNs, CRNNs, CBAM/SE/self-attention variants, MobileNetV2 transfer learning, and vision transformers.
- **Ensemble and optimization tooling** including stacking, soft voting, pruning, quantization, active learning, adversarial training, and Git LFS-backed checkpoints.
- **Advanced analytics** capturing pixel intensity statistics, confusable class clusters, augmentation impact (5× expansion), and actionable deployment recommendations.

## 🧱 Repository Layout

```
.
├── ATTENTION MECHANISM_MODELS/           # CBAM, SE-block, self-attention, ViT checkpoints + histories
├── AUGMENTATION MODEL/                   # Serialized augmentation pipeline for reproducible synthetic sampling
├── Basic_Neural_Network_Models/          # Torch-based dense baselines (.pth)
├── CNN_MODEL_CSV_FILE/                   # Advanced CNN weights and training metrics (.pth/.csv/.pkl)
├── CRNN_MODEL_MULTI_DIGIT_RECOGNITION/   # Sequence-aware CRNN models for multi-digit inference
├── Classical_ML_Models/                  # Pickled scikit-learn pipelines and deployment summaries
├── Datasets_of_Feature_Engineering/      # CSV exports of engineered feature spaces (HOG, LBP, PCA, LDA, etc.)
├── ENSEMBEL_METHODS_MODELS/              # Stacking/soft-voting ensembles with performance snapshots
├── Generative_Models/                    # DCGAN, CGAN, WGAN, VAE, and style transfer generators + metrics
├── Preprocessed_Training_&_Test_Dataset/ # Cached NumPy arrays and pipeline configs for fast reloads
├── Transfer_Learning_Models/             # Fine-tuned MobileNetV2 (Keras/TFLite) and learning histories
├── Complete_Data_Analysis_Report.md      # 70K-sample exploratory analytics and statistical deep dives
├── Challenges_and_Solutions_Report.md    # Engineering retrospective on bottlenecks and mitigations
├── Model_Comparison_Report.md            # Accuracy/latency/size trade-off matrix across approaches
├── PRCP-1002-Handwritten_Digits_Recognition_Final_File3ipynb (1).ipynb
└── PRCP-1002-Handwritten_Digits_Recognition_Final_File3ipynb.json
```

## 🛠️ Environment Setup

1. **Clone and enter the workspace**
   ```bash
   git clone https://github.com/Roy12233444/PRCP-1002-Handwritten-Digits-Recognition.git
   cd PRCP-1002-Handwritten-Digits-Recognition
   ```
2. **Create an isolated environment** (Python ≥3.9 recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Linux/macOS: source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If the requirements file is unavailable, install the core stack:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch torchvision opencv-python xgboost albumentations
   ```
4. **Enable Git LFS** for large model binaries
   ```bash
   git lfs install
   git lfs pull
   ```

## 📚 Working with the Notebook

1. **Launch Jupyter**
   ```bash
   jupyter notebook "PRCP-1002-Handwritten_Digits_Recognition_Final_File3ipynb (1).ipynb"
   ```
2. **Execution flow**
   1. Global configuration and smart NPZ caching
   2. Exploratory analysis and visualization dashboards
   3. Feature engineering pipelines (classical + deep)
   4. Training suites for classical ML, CNNs, attention modules, CRNNs, and transfer learning
   5. Ensemble orchestration and comparative evaluation
   6. Compression experiments (pruning, quantization, TFLite export)
3. **Runtime tips**
   - Set `RANDOM_STATE` consistently to reproduce benchmark tables
   - Use provided `pipeline_config.json` for deterministic preprocessing stages
   - Switch model sections on/off via notebook flags to conserve compute

## 🔄 Data & Preprocessing Pipeline

- **Source dataset**: MNIST (70K grayscale 28×28 digits, balanced across classes).
- **Caching**: NPZ-based loader reduces repeated I/O by ~30%.
- **Normalization**: Pixels scaled to `[0,1]`, optional contrast enhancement.
- **Augmentation**: Rotation (±15°), translation (±10%), scaling (0.9–1.1), shearing, elastic deformation, adaptive histogram equalization, Gaussian noise, Cutout.
- **Feature Engineering**: HOG, LBP, Hu moments, gradient, morphological, Fourier, Wavelet, Zernike, statistical descriptors; PCA (95% variance) and LDA (9 dims) variants provided via CSV exports.

## 🧠 Model Landscape & Benchmarks

| Family | Representative Artifact | Accuracy* | Footprint | Notes |
|---|---|---|---|---|
| Classical ML | `Classical_ML_Models/random_forest_model.pkl` | 94–95% | 20 MB | Uses engineered features + PCA pipeline |
| CNN Baseline | `CNN_MODEL_CSV_FILE/advanced_cnn_chunked_20250706_090829.pth` | 98–99% | 5.45 MB | Adam optimizer, BN, dropout, early stopping |
| Attention CNN (CBAM) | `ATTENTION MECHANISM_MODELS/attention_model_cnn_cbam.h5` | 98% | 1.97 MB | Channel-spatial attention boosts confusable digits |
| Vision Transformer | `ATTENTION MECHANISM_MODELS/attention_model_vision_transformer.h5` | 97–98% | 4.3 MB | Patch embedding + multi-head self-attention |
| CRNN | `CRNN_MODEL_MULTI_DIGIT_RECOGNITION/crnn_model.h5` | 97% | 3.71 MB | Handles sequential multi-digit inputs |
| Transfer Learning | `Transfer_Learning_Models/transfer_learning_mobilenetv2_finetuned_mnist.tflite` | 97% | 2.55 MB | Quantized MobileNetV2 for edge devices |
| Stacking Ensemble | `ENSEMBEL_METHODS_MODELS/stacking_ensemble.h5` | 98.5–99.2% | 2.68 MB | Meta-learner over CNN + dense + specialty models |
| Quantized CNN | `Transfer_Learning_Models/best_transfer_model.h5` (pre-quant) | 94.8% | 230 KB (TFLite) | 73% size reduction with minimal accuracy loss |

*Accuracy ranges derive from `Model_Comparison_Report.md` and notebook evaluations; rerun cells to reproduce figures under your hardware constraints.

## 📈 Evaluation & Diagnostics

- **Confusion analysis**: Top confusions (3↔8, 4↔9, 6↔8) visualized with per-class metrics in the notebook.
- **Dimensionality insights**: PCA retains 95% variance within 150–200 components; t-SNE plots highlight cluster separability.
- **Quality metrics**: Edge density, brightness, contrast, and sparsity tracked per digit to inform augmentation focus.
- **Reports**: Deep dives available in `Complete_Data_Analysis_Report.md`, `Challenges_and_Solutions_Report.md`, and `Model_Comparison_Report.md`.

## 🚢 Deployment Playbook

1. **Baseline export**: Save best Keras/Torch weights during training checkpoints.
2. **Compression**: Apply pruning (80% sparsity) followed by post-training quantization for 4×–5× footprint reduction with <5% accuracy drop.
3. **Edge delivery**: Use bundled `.tflite` artifacts; integrate with TensorFlow Lite `Interpreter` for <10 ms inference on mobile-class CPUs.
4. **Monitoring**: Track confusable pairs and drift via periodic evaluation on augmented validation sets.

## ♻️ Reproducibility Checklist

1. Activate the virtual environment and install dependencies.
2. Pull LFS artifacts to ensure models and datasets are available locally.
3. Execute notebook sections sequentially, seeding `numpy`, `tensorflow`, and `torch` with the same `RANDOM_STATE`.
4. Use cached NumPy datasets and feature CSVs for deterministic classical model training.
5. Record checkpoints and evaluation metrics for comparison against the supplied reports.

## 🤝 Contributing

- **Issues**: Use GitHub Issues for bug reports, feature requests, or clarifications.
- **Pull Requests**: Provide reproducible benchmarks and mention affected artifacts (models, datasets, reports).
- **Experiments**: New architectures should include validation metrics and updated analysis snapshots.

## 📄 License & Attribution

- **License**: MIT — consult `LICENSE` for terms.
- **Dataset**: MNIST by Yann LeCun et al. (public domain). Please cite appropriately in derivative work.
