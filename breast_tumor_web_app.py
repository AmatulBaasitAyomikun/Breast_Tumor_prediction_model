# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:08:50 2025

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Function for prediction
def breast_tumor_prediction(input_data):
    input_data_array = np.asarray(input_data)
    input_data_reshaped = input_data_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return '🔴 The Tumor is Malignant'
    else:
        return '🟢 The Tumor is Benign'

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Breast Tumor Prediction",
        page_icon="🎗️",
        layout="centered"
    )

    st.markdown(
        """
        <h2 style="text-align:center;">🎗️ Breast Tumor Prediction Web App</h2>
        <p style="text-align:center; color:gray;">
        Enter the tumor measurements below to predict whether it is benign or malignant.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("---")
    
    # -------- Mean Features --------
    st.subheader("Mean Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_mean = st.number_input("radius_mean")
        perimeter_mean = st.number_input("perimeter_mean")
        area_mean = st.number_input("area_mean")
        concavity_mean = st.number_input("concavity_mean")
        symmetry_mean = st.number_input("symmetry_mean")
    with col2:
        texture_mean = st.number_input("texture_mean")
        smoothness_mean = st.number_input("smoothness_mean")
        compactness_mean = st.number_input("compactness_mean")
        concave_points_mean = st.number_input("concave_points_mean")
        fractal_dimension_mean = st.number_input("fractal_dimension_mean")
    with col3:
        st.write("")  # empty column for spacing
        st.write("")
        st.write("")
        st.write("")
        st.write("")

    st.write("---")

    # -------- SE Features --------
    st.subheader("Standard Error (SE) Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_se = st.number_input("radius_se")
        perimeter_se = st.number_input("perimeter_se")
        area_se = st.number_input("area_se")
        concavity_se = st.number_input("concavity_se")
        symmetry_se = st.number_input("symmetry_se")
    with col2:
        texture_se = st.number_input("texture_se")
        smoothness_se = st.number_input("smoothness_se")
        compactness_se = st.number_input("compactness_se")
        concave_points_se = st.number_input("concave_points_se")
        fractal_dimension_se = st.number_input("fractal_dimension_se")
    with col3:
        st.write("")  # empty for spacing
        st.write("")
        st.write("")
        st.write("")
        st.write("")

    st.write("---")

    # -------- Worst Features --------
    st.subheader("Worst Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_worst = st.number_input("radius_worst")
        perimeter_worst = st.number_input("perimeter_worst")
        area_worst = st.number_input("area_worst")
        concavity_worst = st.number_input("concavity_worst")
        symmetry_worst = st.number_input("symmetry_worst")
    with col2:
        texture_worst = st.number_input("texture_worst")
        smoothness_worst = st.number_input("smoothness_worst")
        compactness_worst = st.number_input("compactness_worst")
        concave_points_worst = st.number_input("concave_points_worst")
        fractal_dimension_worst = st.number_input("fractal_dimension_worst")
    with col3:
        st.write("")  # empty for spacing
        st.write("")
        st.write("")
        st.write("")
        st.write("")

    st.write("---")

    # -------- Prediction Button --------
    diagnosis = ""
    if st.button("Predict Tumor Type"):
        diagnosis = breast_tumor_prediction([
            radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean,
            concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se,
            smoothness_se, compactness_se, concavity_se,
            concave_points_se, symmetry_se, fractal_dimension_se,
            radius_worst, texture_worst, perimeter_worst, area_worst,
            smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst
        ])
        st.success(diagnosis)

    st.write("---")
    st.markdown(
        "<p style='text-align:center; color:gray;'>Built with ❤️ using Machine Learning & Streamlit</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    