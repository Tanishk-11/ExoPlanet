import streamlit as st
import pandas as pd
import joblib
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.set_page_config(page_title="Exoplanet Classifier", layout="wide")

def set_styles():
    try:
        img_base64 = get_base64_of_bin_file("background.jpg")
        page_bg_img = f"""
        <style>
        /* --- General & Background --- */
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* --- Text Styling --- */
        h1, h2, h3, p, label, .st-emotion-cache-1tplte7 p {{
            color: white;
        }}
        
        /* --- Container Styling --- */
        div[data-testid="stForm"], div[data-testid="stExpander"] {{
             background-color: rgba(0, 0, 0, 0.7);
             border-radius: 10px;
             padding: 20px;
        }}
        
        /* --- Button Styling --- */
        div[data-testid="stForm"] button {{
            background-color: #008CBA;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 24px;
        }}
        div[data-testid="stForm"] button:hover {{
            background-color: #007B9A;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Background image 'background.jpg' not found. App will have a dark theme.")

set_styles()

@st.cache_resource
def load_model_files():
    try:
        gmm = joblib.load('gmm_model_1.joblib')
        scaler = joblib.load('scaler.joblib')
        profiles = pd.read_csv('cluster_profiles.csv', index_col='Cluster')
        return gmm, scaler, profiles
    except FileNotFoundError:
        return None, None, None

gmm, scaler, profiles = load_model_files()

def interpret_cluster_profile(profile):
    mass = profile['Mass']
    radius = profile['Radius']
    temp = profile['Temperature']
    period = profile['Orbital Period']
    
    interpretation = "This cluster represents a diverse group of mid-sized planets."
    if mass > 100 and temp > 1000 and period < 10:
        interpretation = "This cluster represents 'Hot Jupiters'—massive gas giants orbiting extremely close to their stars."
    elif mass > 50:
        interpretation = "This cluster represents 'Gas Giants,' similar in scale to Jupiter or Saturn."
    elif mass > 2 and radius > 1.5:
        interpretation = "This cluster represents 'Super-Earths' or 'Mini-Neptunes'—planets larger than Earth but smaller than Neptune."
    elif mass < 2 and radius < 1.5:
        interpretation = "This cluster represents small, rocky worlds, potentially similar to Earth or Mars."

    habitability = "The average temperature of this group is outside the traditional habitable zone."
    if 273 <= temp <= 373:
        habitability = "Notably, the average temperature of this group is in the 'Habitable Zone,' where liquid water could potentially exist."
        
    return interpretation, habitability

st.title("🪐 Exoplanet Classifier")
st.write("Enter the properties of a planet to predict which cluster it belongs to and see its likely characteristics.")

if gmm is None:
    st.error("Model files not found! Please ensure gmm_model.joblib, scaler.joblib, and cluster_profiles.csv are in the same folder.")
else:
    with st.form("planet_form"):
        st.header("Input Planet Data")
        
        col1, col2 = st.columns(2)
        with col1:
            mass = st.number_input("Mass (in Earth Masses)", min_value=0.1, value=1.0, step=0.1)
            radius = st.number_input("Radius (in Earth Radii)", min_value=0.1, value=1.0, step=0.1)
            temp = st.number_input("Equilibrium Temperature (K)", min_value=1, value=300, step=10)
        
        with col2:
            period = st.number_input("Orbital Period (days)", min_value=0.1, value=365.0, step=1.0)
            stellar_temp = st.number_input("Host Star Temperature (K)", min_value=1000, value=5700, step=100)
        
        submitted = st.form_submit_button("Classify Planet")

    if submitted:
        features = ['Mass', 'Radius', 'Temperature', 'Orbital Period', 'Stellar Temperature']
        planet_data = {'Mass': mass, 'Radius': radius, 'Temperature': temp, 'Orbital Period': period, 'Stellar Temperature': stellar_temp}
        
        new_planet_df = pd.DataFrame([planet_data], columns=features)
        new_planet_scaled = scaler.transform(new_planet_df)
        
        predicted_cluster = gmm.predict(new_planet_scaled)[0]
        cluster_profile = profiles.loc[predicted_cluster]
        
        interpretation, habitability = interpret_cluster_profile(cluster_profile)
        
        st.header("Prediction Result")
        st.success(f"This new planet belongs to **Cluster {predicted_cluster}**.")
        
        st.subheader("Cluster Interpretation")
        st.info(f"**Interpretation:** {interpretation}\n\n**Habitability Insight:** {habitability}")
        
        # --- The New, Corrected Display Method ---
        with st.expander("Show Cluster Profile (Average Values)"):
            cols = st.columns(5)
            cols[0].metric("Mass (Earths)", f"{cluster_profile['Mass']:.2f}")
            cols[1].metric("Radius (Earths)", f"{cluster_profile['Radius']:.2f}")
            cols[2].metric("Temperature (K)", f"{cluster_profile['Temperature']:.0f}")
            cols[3].metric("Orbital Period (days)", f"{cluster_profile['Orbital Period']:.2f}")
            cols[4].metric("Stellar Temperature (K)", f"{cluster_profile['Stellar Temperature']:.0f}")