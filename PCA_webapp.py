import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to perform PCA
def perform_pca(data, num_components):
    # Standardize the data
    standardized_data = StandardScaler().fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(standardized_data)

    # Create a DataFrame with the principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(num_components)])

    return pc_df, pca.explained_variance_ratio_

# Streamlit App
def main():
    st.title("PCA Web App")

    # Upload dataset
    uploaded_file = st.file_uploader("Choose your csv file", type="csv")

    if uploaded_file is not None:
        st.sidebar.header('Settings')

        # Read data
        data = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader('Raw Data')
        st.write(data.head())

        # Perform PCA
        num_components = st.sidebar.slider("Select number of components", 1, min(data.shape), 2)
        pc_df, explained_variance_ratio = perform_pca(data, num_components)

        # Display explained variance ratio
        st.sidebar.subheader('Explained Variance Ratio')
        st.sidebar.text(explained_variance_ratio)

        # Display the principal components
        st.subheader('Principal Components')
        st.write(pc_df.head())

        # Plot explained variance ratio
        st.sidebar.subheader('Explained Variance Ratio Plot')
        st.sidebar.bar_chart(explained_variance_ratio)

if __name__ == '__main__':
    main()
