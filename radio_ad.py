# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""# radio_ads_app.py

# radio_ads_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Define allowed variables

DVs = ['Stands out',
 'I would listen to it',
 "It's clear who it is for",
 'Advertising I would remember',
 'Speaks my language',
 'Annoying',
 'Makes you feel more positive',
 'Informative',
 'Clear and easy to follow']

X_vars_allowed = ['Involvement',
 'Identity', 
 'Impression', 
 'Information', 
 'Integration',
 'Recognise catchphrase/slogan',
 'Recognise music/voice',
 'Actors same No',
 'Voice TV Yes',
 'Voice TV No',
 'Music Yes',
 'Music No',
 'Music same Yes',
 'Music same No',
 'Music TV link Yes',
 'Music TV link No',
 'Sonic brand device Yes',
 'Sonic brand device No',
 'Recognisable Strapline Yes',
 'Recognisable Strapline No',
 'Brand Character Featured Yes',
 'Brand Character Featured No',
 'Story/Scene Integrated Message',
 'Straightforward Announcement',
 'Brand Reveal Start',
 'Brand Reveal Middle',
 'Brand Reveal End',
 'Humour Yes',
 'Homour No',
 'Punchline Yes',
 'Punchline No',
 'CTA Yes',
 'CTA No',
 'Call to action In Store',
 'Call to action Telephone',
 'Call to action Website',
 'Call to action Online',
 'Call to action Search',
 'Press',
 'Radio',
 'TV',
 'Radio integrated with TV Yes',
 'Radio integrated with TVNo',
 'T&Cs Excessive',
 'T&Cs Minimal',
 'T&Cs None',
 'Recognisable voice No', 'valence', 'arousal', 'dominance', 'auditory', 'gustatory', 'interoceptive', 'olfactory', 
 'visual', 'foot_leg', 'hand_arm', 'head', 'mouth', 'torso',
 'concreteness', 'imageability', 'semantic_size', 'haptic']

# App Title
st.title("Ridge Regression Explorer — Radio Ads Data")

# Upload data
uploaded_file = st.file_uploader("Upload Final Cleaned Data (final_cleaned_data.xlsx)", type=["xlsx"])

if uploaded_file:
    # Load data
    data = pd.read_excel(uploaded_file, index_col=0)
    st.write("Initial Data Preview:", data.head())

    # Drop NAs
    data = data.dropna()
    st.write(f"Data after dropna(): {data.shape[0]} rows, {data.shape[1]} columns")

    # Variable selection — restricted lists
    y_var = st.selectbox("Select Dependent Variable (Output Metric)", [var for var in DVs if var in data.columns])
    x_vars = st.multiselect("Select Independent Variables (Predictors)", [var for var in X_vars_allowed if var in data.columns])

    if y_var and x_vars:
        # Prepare X and y
        X = data[x_vars]
        y = data[y_var]

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Build pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5))
        ])

        # Fit model
        pipeline.fit(X, y)
        ridge_model = pipeline.named_steps['ridge']

        # Cross-validated R²
        cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        st.write(f"Mean CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

        # Coefficients
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': ridge_model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=False)

        st.write("Ridge Coefficients:")
        st.dataframe(coef_df)

        # Plot coefficients
        fig, ax = plt.subplots(figsize=(10, len(coef_df) * 0.3))
        ax.barh(coef_df['Feature'], coef_df['Coefficient'])
        ax.set_xlabel('Ridge Coefficient')
        ax.set_title(f'Feature Importance for "{y_var}" (Ridge Regression)')
        ax.grid(True)
        st.pyplot(fig)

        # Download button
        csv = coef_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Coefficients as CSV",
            data=csv,
            file_name='ridge_coefficients.csv',
            mime='text/csv'
        )




