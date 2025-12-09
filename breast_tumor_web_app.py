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

# creating a function for prediction

def breast_tumor_prediction(input_data):
    
    input_data_array = np.asarray(input_data)

    input_data_reshaped = input_data_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if (prediction[0] == 0):
      return 'The Tumor is Malignant'

    else:
      return 'The Tumor is Benign'
  
    
# creating streamlit interface

def main():
        st.title('Breast Tumor Prediction Web App')
        
        radius_mean = st.number_input("radius_mean")
        texture_mean = st.number_input( "texture_mean")
        perimeter_mean = st.number_input("perimeter_mean")
        area_mean = st.number_input("area_mean")
        smoothness_mean = st.number_input("smoothness_mean")
        compactness_mean = st.number_input("compactness_mean")
        concavity_mean = st.number_input("concavity_mean")
        concave_points_mean = st.number_input("concave points_mean")
        symmetry_mean = st.number_input("symmetry_mean")
        fractal_dimension_mean = st.number_input("fractal_dimension_mean")
        radius_se = st.number_input("radius_se")
        texture_se = st.number_input("texture_se")
        perimeter_se = st.number_input("perimeter_se")
        area_se = st.number_input("area_se")
        smoothness_se = st.number_input("smoothness_se")
        compactness_se = st.number_input("compactness_se")
        concavity_se = st.number_input("concavity_se")
        concave_points_se = st.number_input("concave points_se")
        symmetry_se = st.number_input("symmetry_se")
        fractal_dimension_se = st.number_input("fractal_dimension_se")
        radius_worst = st.number_input("radius_worst")
        texture_worst = st.number_input("texture_worst")
        perimeter_worst = st.number_input("perimeter_worst")
        area_worst = st.number_input("area_worst")
        smoothness_worst = st.number_input("smoothness_worst")
        compactness_worst = st.number_input("compactness_worst")
        concavity_worst = st.number_input("concavity_worst")
        concave_points_worst = st.number_input("concave points_worst")
        symmetry_worst = st.number_input("symmetry_worst")
        fractal_dimension_worst = st.number_input("fractal_dimension_worst")
          
    
    #code for prediction
        diagnosis = " "

        if st.button('Confirm Breast Tumor Type'):
            diagnosis = breast_tumor_prediction(radius_mean, texture_mean, perimeter_mean, area_mean,
                   smoothness_mean, compactness_mean, concavity_mean,
                   concave_points_mean, symmetry_mean, fractal_dimension_mean,
                   radius_se, texture_se, perimeter_se, area_se,
                   smoothness_se, compactness_se, concavity_se,
                   concave_points_se, symmetry_se, fractal_dimension_se,
                   radius_worst, texture_worst, perimeter_worst, area_worst,
                   smoothness_worst, compactness_worst, concavity_worst,
                   concave_points_worst, symmetry_worst, fractal_dimension_worst)
    
        st.success(diagnosis)
        
        
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    