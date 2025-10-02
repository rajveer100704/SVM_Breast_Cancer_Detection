# Task 7: Support Vector Machines (SVM) on Breast Cancer Dataset

## Objective
Classify breast cancer cases into **malignant** or **benign** using Support Vector Machines (SVM) with linear and RBF kernels. Perform hyperparameter tuning and visualize decision boundaries.

## Dataset
- [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)  
- Features include measurements of cell nuclei from biopsy images.  
- Target: `diagnosis` (`M` = malignant, `B` = benign)

## Steps Performed
1. **Data Preprocessing**
   - Loaded dataset and mapped target variable to numeric (`M=1`, `B=0`)
   - Standardized features using `StandardScaler`
   - Split data into training and test sets

2. **SVM Training**
   - Trained **Linear SVM** for baseline performance
   - Trained **RBF kernel SVM** for non-linear separation

3. **Dimensionality Reduction**
   - Used **PCA** to reduce features to 2D for visualization of decision boundaries

4. **Hyperparameter Tuning**
   - Applied `GridSearchCV` to find the best `C` and `gamma` for RBF SVM
   - Best parameters: `C=1`, `gamma='scale'`, `kernel='rbf'`
   - Achieved cross-validation accuracy ≈ **97.6%**

5. **Visualization**
   - Plotted decision boundaries in 2D PCA space for optimized SVM
   - Visualized how the model separates malignant and benign cases

6. **Evaluation**
   - Evaluated using **confusion matrix** and **classification report**
   - Optimized RBF SVM accuracy: **97%** on test set

## Libraries Used
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run
1. Clone the repository
2. Open `breast_cancer_svm.ipynb` in Google Colab
3. Upload `kaggle.json` for Kaggle API access
4. Run all cells sequentially

## Files
- `breast_cancer_svm.ipynb` – Complete Colab notebook with code, visualizations, and results
- `data/` – Folder containing the dataset CSV
- `README.md` – This file

## Key Learnings
- SVM separates data using **maximum margin** hyperplanes
- RBF kernel allows **non-linear decision boundaries**
- Hyperparameter tuning (`C` and `gamma`) improves performance
- PCA helps **visualize high-dimensional data** in 2D
