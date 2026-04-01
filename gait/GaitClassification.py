"""
Gait phase classification for multi-limb locomotion analysis.

This module trains and evaluates a multi-output classifier to predict gait phases
(swing/stance/unknown) for each limb independently using kinematic features extracted
from pose tracking data. It uses LightGBM as the base estimator with scikit-learn's
MultiOutputClassifier to handle simultaneous prediction across all limbs.

Workflow
--------
1. Load CSV with kinematic features and limb labels (ForepawR, ForepawL, HindpawR, HindpawL)
2. Encode categorical labels (swing/stance/unknown) to numeric format
3. Train a multi-output LightGBM classifier on 80% of data
4. Evaluate performance on held-out 20% test set
5. Generate diagnostic outputs:
   - Per-limb classification reports and confusion matrices
   - Overall Hamming loss and subset accuracy metrics
   - Visualisations of misclassified samples with predicted vs true labels
   - Feature importance rankings

Outputs
-------
- limb_classification_model.pkl : Trained multi-output classifier
- label_encoders.pkl : Label encoders for decoding predictions
- feature_columns.pkl : List of features used for training
- confusion_matrix_<limb>.png : Per-limb confusion matrices
- feature_importance.png : Top 20 most important features
- Misclassified_Plots/ : Images of misclassified frames with annotations

Notes
-----
- Requires pre-extracted features CSV with 'Filename' and 'Frame' columns
- Expects corresponding video frames in structured directories for visualisation
- Handles 'unknown' labels as a valid class (e.g., for occluded limbs)
- Uses MultiOutputClassifier to maintain independence between limb predictions

"""

import pandas as pd
import os
import glob
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib


def load_and_preprocess_data(data_path, label_columns):
    data = pd.read_csv(data_path)

    # Prepare feature columns (exclude label columns and non-feature columns)
    non_feature_columns = ['Filename', 'Frame'] + label_columns
    feature_columns = [col for col in data.columns if col not in non_feature_columns]

    # Keep 'Filename' and 'Frame' in X for later use
    X = data[feature_columns + ['Filename', 'Frame']]
    y = data[label_columns]

    # Ensure all 'unknown' labels are consistent
    y = y.replace({'unknown': 'unknown', np.nan: 'unknown'})

    # Encode labels, including 'unknown' as a class
    label_encoders = {}
    for col in label_columns:
        label_enc = LabelEncoder()
        y[col] = label_enc.fit_transform(y[col].astype(str))
        label_encoders[col] = label_enc
        class_names = label_enc.classes_
        print(f"Classes for {col}:", class_names)

    # Reset indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    return X, y, feature_columns, label_encoders


def train_model(X_train_features, y_train, output_dir, model_type='lightgbm'):
    if model_type == 'lightgbm':
        base_estimator = lgb.LGBMClassifier(
            objective='multiclass',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            verbose=-1,
            n_jobs=-1
        )
    else:
        # Add other model types if needed
        pass

    # Create the multi-output classifier
    multi_target_model = MultiOutputClassifier(base_estimator, n_jobs=-1)

    # Train the model
    multi_target_model.fit(X_train_features, y_train)

    # Save the model to disk
    model_filename = f'limb_classification_model.pkl'
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(multi_target_model, model_path)
    print(f"Model saved as {model_filename}")

    return multi_target_model


def multiclass_multioutput_hamming_loss(y_true, y_pred):
    """
    Compute the Hamming loss for multiclass-multioutput classification.

    Parameters:
    y_true (numpy array or pandas DataFrame): True labels of shape (n_samples, n_outputs)
    y_pred (numpy array): Predicted labels of shape (n_samples, n_outputs)

    Returns:
    float: Hamming loss
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Total number of labels
    n_samples, n_outputs = y_true.shape
    total_labels = n_samples * n_outputs

    # Number of mismatches
    mismatches = (y_true != y_pred).sum()

    # Hamming loss
    hamming_loss = mismatches / total_labels
    return hamming_loss


def plot_misclassified_sample(base_directory, filename, frame_num, true_labels, predicted_labels, output_dir, label_encoders):
    # Construct the image path
    subdir = filename.replace('_mapped3D.h5', '')
    image_filename_png = f'img{frame_num}.png'
    image_path_pattern = os.path.join(base_directory, subdir + '*')
    matching_dirs = glob.glob(image_path_pattern)

    if matching_dirs:
        image_dir = matching_dirs[0]
        image_path = os.path.join(image_dir, image_filename_png)
    else:
        print(f"Image directory not found for filename {filename}")
        return

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Load the image
    image = plt.imread(image_path)

    # Plot the image
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    plt.axis('off')

    # Prepare the label text using label_encoders to decode labels
    true_label_text_list = []
    predicted_label_text_list = []
    for limb in true_labels.keys():
        label_enc = label_encoders[limb]
        true_label_decoded = label_enc.inverse_transform([true_labels[limb]])[0]
        predicted_label_decoded = label_enc.inverse_transform([predicted_labels[limb]])[0]
        true_label_text_list.append(f"{limb}: {true_label_decoded}")
        predicted_label_text_list.append(f"{limb}: {predicted_label_decoded}")

    true_label_text = '\n'.join(true_label_text_list)
    predicted_label_text = '\n'.join(predicted_label_text_list)

    # Add text annotations outside the image
    plt.gcf().text(0.05, 0.95, f"True Labels:\n{true_label_text}", fontsize=10, color='green', verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.7))
    plt.gcf().text(0.05, 0.75, f"Predicted Labels:\n{predicted_label_text}", fontsize=10, color='red', verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.7))

    plt.title(f"Misclassified Sample\nFilename: {filename}, Frame: {frame_num}")

    # Save the plot to a file
    os.makedirs(os.path.join(output_dir, "Misclassified_Plots"), exist_ok=True)
    plot_filename = f"{filename}_frame{frame_num}_misclassified_{limb}.png"
    plot_path = os.path.join(output_dir, "Misclassified_Plots", plot_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory


def analyze_misclassifications(X_test, y_test, y_pred, label_columns, base_directory, output_dir, label_encoders):
    for idx, col in enumerate(label_columns):
        print(f"\nAnalyzing misclassifications for {col}:")

        # Compare y_test values with predictions directly
        misclassified = y_test[col].values != y_pred[:, idx]
        num_misclassified = misclassified.sum()
        print(f'Number of misclassified samples: {num_misclassified}')

        if num_misclassified > 0:
            # Extract misclassified samples
            X_misclassified = X_test.iloc[misclassified].reset_index(drop=True)
            y_true_misclassified = y_test.iloc[misclassified].reset_index(drop=True)
            y_pred_misclassified = y_pred[misclassified]

            # For each misclassified sample, plot and save the image
            for i in range(len(X_misclassified)):
                filename = X_misclassified.loc[i, 'Filename']
                frame_num = X_misclassified.loc[i, 'Frame']

                # Prepare labels as dictionaries
                true_labels = {col: y_true_misclassified[col].iloc[i]}
                predicted_labels = {col: y_pred_misclassified[i, idx]}

                # Plot and save the misclassified sample
                plot_misclassified_sample(
                    base_directory, filename, frame_num,
                    true_labels, predicted_labels,
                    output_dir, label_encoders
                )


def plot_feature_importance(model, feature_columns, output_dir):
    # Since we're using MultiOutputClassifier, we can average importances
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(20, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))  # Plot top 20 features
    plt.title('Top 20 Feature Importance')
    plt.yticks(fontsize=8)

    # Save the plot
    plot_filename = 'feature_importance.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()


def main():
    # Define data path and label columns
    data_path = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff\extracted_features.csv"
    output_dir = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff"
    label_columns = ['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']

    # Load and preprocess data
    X, y, feature_columns, label_encoders = load_and_preprocess_data(data_path, label_columns)

    # Save label encoders and feature columns
    label_encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, label_encoders_path)
    print(f"Label encoders saved as label_encoders.pkl")

    feature_columns_path = os.path.join(output_dir, 'feature_columns.pkl')
    joblib.dump(feature_columns, feature_columns_path)
    print(f"Feature columns saved as feature_columns.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Reset indices
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Prepare feature matrices for training and testing
    X_train_features = X_train[feature_columns]
    X_test_features = X_test[feature_columns]

    # Train the model
    multi_target_model = train_model(X_train_features, y_train, output_dir, model_type='lightgbm')


    # Predict on test data
    y_pred = multi_target_model.predict(X_test_features)

    # Evaluate the model
    print("Model Evaluation:")
    for idx, col in enumerate(label_columns):
        print(f"\nEvaluation for {col}:")
        accuracy = accuracy_score(y_test[col], y_pred[:, idx])
        print(f'Accuracy: {accuracy:.4f}')
        print('Classification Report:')
        print(classification_report(y_test[col], y_pred[:, idx], target_names=label_encoders[col].classes_))
        print('-' * 50)

        # Confusion Matrix
        cm = confusion_matrix(y_test[col], y_pred[:, idx])
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders[col].classes_)
        cm_display.plot()
        plt.title(f'Confusion Matrix for {col}')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{col}.png'))
        plt.close()

    # Calculate overall hamming loss using the custom function
    hloss = multiclass_multioutput_hamming_loss(y_test, y_pred)
    print(f'Overall Hamming Loss: {hloss:.4f}')

    # Optionally, calculate subset accuracy (exact match ratio)
    subset_accuracy = (y_pred == y_test.values).all(axis=1).mean()
    print(f'Subset Accuracy (Exact Match Ratio): {subset_accuracy:.4f}')

    # Analyze misclassifications and save images
    base_directory = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side"
    analyze_misclassifications(X_test, y_test, y_pred, label_columns, base_directory, output_dir, label_encoders)

    # Plot feature importance
    plot_feature_importance(multi_target_model, feature_columns, output_dir)

    plt.show()


if __name__ == "__main__":
    main()
