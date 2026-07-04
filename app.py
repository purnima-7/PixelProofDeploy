import streamlit as st

st.title("Debug App")

try:
    import joblib
    st.success("✅ joblib imported")
except Exception as e:
    st.exception(e)
    st.stop()

st.success("🎉 App is working!")