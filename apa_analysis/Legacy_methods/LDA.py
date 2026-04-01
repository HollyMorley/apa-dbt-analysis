import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def perform_lda(pcs, y_labels, phase1, phase2, n_components):
    """
    Perform LDA on the PCA-transformed data.
    """
    lda = LDA(n_components=n_components)
    lda.fit(pcs, y_labels)
    Y_lda = lda.transform(pcs)
    lda_loadings = lda.coef_[0]
    return lda, Y_lda, lda_loadings


def compute_feature_contributions(loadings_df, lda_loadings):
    """
    Compute original feature contributions to LDA.
    """
    original_feature_contributions = loadings_df.dot(lda_loadings)
    feature_contributions_df = pd.DataFrame({
        'Feature': original_feature_contributions.index,
        'Contribution': original_feature_contributions.values
    })
    return feature_contributions_df


def plot_feature_contributions(feature_contributions_df, save_path, title_suffix=""):
    """
    Plot the LDA feature contributions.
    """
    plt.figure(figsize=(14, 50))
    sns.barplot(data=feature_contributions_df, x='Contribution', y='Feature', palette='viridis')
    plt.title(f'Original Feature Contributions to LDA Component {title_suffix}')
    plt.xlabel('Contribution to LDA')
    plt.ylabel('Original Features')
    plt.axvline(0, color='red', linewidth=1)
    plt.tight_layout()
    plot_filename = "Feature_Contributions_to_LDA"
    if title_suffix:
        plot_filename += f"_{title_suffix.replace(' ', '_')}"
    plot_filename += ".png"
    plt.savefig(os.path.join(save_path, plot_filename), dpi=300)
    plt.close()


def plot_lda_transformed_data(Y_lda, phase1, phase2, save_path, title_suffix=""):
    """
    Plot the LDA transformed data (scatter and box plots).
    """
    num_phase1 = int(len(Y_lda) / 2)
    num_phase2 = len(Y_lda) - num_phase1
    df_lda = pd.DataFrame(Y_lda, columns=['LDA_Component'])
    df_lda['Condition'] = [phase1] * num_phase1 + [phase2] * num_phase2

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_lda, x='LDA_Component', y=[0] * df_lda.shape[0],
                    hue='Condition', style='Condition', s=100, alpha=0.7)
    plt.title(f'LDA Transformed Data for {phase1} vs {phase2} ({title_suffix})')
    plt.xlabel('LDA Component')
    plt.yticks([])
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True)
    plt.tight_layout()
    scatter_plot_filename = f"LDA_{phase1}_vs_{phase2}_{title_suffix.replace(' ', '_')}_Scatter.png"
    plt.savefig(os.path.join(save_path, scatter_plot_filename), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Condition', y='LDA_Component', data=df_lda, palette='Set2')
    plt.title(f'LDA Component Distribution by Condition ({phase1} vs {phase2}) ({title_suffix})')
    plt.xlabel('Condition')
    plt.ylabel('LDA Component')
    plt.grid(True)
    plt.tight_layout()
    box_plot_filename = f"LDA_{phase1}_vs_{phase2}_{title_suffix.replace(' ', '_')}_Box.png"
    plt.savefig(os.path.join(save_path, box_plot_filename), dpi=300)
    plt.close()

def plot_LDA_loadings(lda_loadings_all, save_path, title_suffix=""):
    # Optional LDA loadings bar plot.
    plt.figure(figsize=(12, 6))
    pc_indices_all = [f'PC{i + 1}' for i in range(len(lda_loadings_all))]
    plt.bar(pc_indices_all, lda_loadings_all, color='skyblue')
    plt.title(f'LDA Loadings on PCA Components {title_suffix}')
    plt.xlabel('Principal Components')
    plt.ylabel('LDA Coefficients')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"LDA_Loadings_on_PCA_Components_All_{title_suffix}.png"), dpi=300)
    plt.close()