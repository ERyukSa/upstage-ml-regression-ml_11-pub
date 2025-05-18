import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_transportation_features(metrics, X_train):
    """
    Analyze the importance of transportation features in the model
    
    Args:
        metrics (dict): Dictionary containing model metrics including feature importances
        X_train (DataFrame): Training dataset with features
        
    Returns:
        None
    """
    print("\n===== TRANSPORTATION FEATURE IMPORTANCE ANALYSIS =====")
    
    # Get feature importances
    feature_importances = metrics['feature_importances']
    
    # Fix: Print available columns to diagnose issues
    print("Available columns in feature_importances:", feature_importances.columns.tolist())
    
    # Fix: Use the 'Importance' column directly since we know it exists
    importance_col = 'Importance'
    
    # Identify transportation features (expanded list with 'hub')
    transportation_cols = [col for col in X_train.columns if any(term in col for term in 
                                                                ['subway', 'bus', 'transit', 'hub'])]
    
    if not transportation_cols:
        print("No transportation features found in the dataset.")
        return
    
    print(f"Found {len(transportation_cols)} transportation-related features:")
    print(", ".join(transportation_cols))
    
    # Extract transportation feature importances
    transport_importances = feature_importances[feature_importances['Feature'].isin(transportation_cols)]
    
    # Print importance of transportation features
    print("\nTransportation Feature Importances:")
    if len(transport_importances) > 0:
        transport_importances = transport_importances.sort_values(importance_col, ascending=False)
        for _, row in transport_importances.iterrows():
            print(f"- {row['Feature']}: {row[importance_col]:.4f}")
        
        # Calculate percent of total importance from transportation features
        total_importance = feature_importances[importance_col].sum()
        transport_importance = transport_importances[importance_col].sum()
        importance_percent = (transport_importance / total_importance) * 100
        
        print(f"\nTransportation features contribute {importance_percent:.2f}% of total feature importance")
        
        # Visualize top transportation features
        plt.figure(figsize=(12, 6))
        
        # Plot top 10 transportation features or all if less than 10
        num_features = min(10, len(transport_importances))
        top_transport = transport_importances.head(num_features)
        
        sns.barplot(x=importance_col, y='Feature', data=top_transport)
        plt.title('Top Transportation Feature Importances')
        plt.xlabel('Importance (Gain)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Compare with other feature categories
        print("\nComparing with other feature categories:")
        
        # Define feature categories
        categories = {
            'Transportation': transportation_cols,
            'Area_related': [col for col in X_train.columns if any(term in col for term in 
                                                                  ['전용면적', 'area', 'size'])],
            'Building_age': [col for col in X_train.columns if any(term in col for term in 
                                                                  ['건축년도', 'building_age'])],
            'Floor': [col for col in X_train.columns if any(term in col for term in 
                                                          ['층', 'floor'])],
            'Location': [col for col in X_train.columns if any(term in col for term in 
                                                             ['구', '동', 'district', 'location'])]
        }
        
        # Calculate importance by category
        category_importance = {}
        for category, cols in categories.items():
            # Get features that exist in the feature importances
            existing_cols = [col for col in cols if col in feature_importances['Feature'].values]
            importance = feature_importances[feature_importances['Feature'].isin(existing_cols)][importance_col].sum()
            category_importance[category] = importance
        
        # Visualize category importance
        plt.figure(figsize=(10, 6))
        categories_df = pd.DataFrame({
            'Category': list(category_importance.keys()),
            'Importance': list(category_importance.values())
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Category', data=categories_df)
        plt.title('Feature Importance by Category')
        plt.xlabel('Total Importance (Gain)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("None of the transportation features were used significantly by the model.")