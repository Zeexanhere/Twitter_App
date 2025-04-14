import streamlit as st
import pickle
import time
import numpy as np

# Load image
st.image("Twitter.jpg", width=620)

# Title
st.title("Twitter Sentiment Analysis")

# Load model
model = pickle.load(open('twitter_sentiment.pkl', 'rb'))

# Text input
tweet = st.text_input("Enter Your Tweet here")

# Sentiment to Emoji mapping
emoji_map = {
    "positive": "😊",
    "negative": "😠",
    "neutral": "😐"
}

if st.button("Predict"):
    if tweet:
        with st.spinner("Analyzing tweet..."):
            start_time = time.time()
            prediction = model.predict([tweet])[0]
            end_time = time.time()
            execution_time = end_time - start_time

            # Try to get confidence score
            try:
                proba = model.predict_proba([tweet])
                confidence = np.max(proba) * 100
                st.metric("Confidence Score", f"{confidence:.2f}%")
            except:
                st.warning("Confidence score not available for this model.")
                confidence = None

            # Display sentiment with emoji
            emoji = emoji_map.get(prediction.lower(), "🤔")
            st.success(f"Sentiment: **{prediction.upper()}** {emoji}")
            st.caption(f"Execution Time: {execution_time:.4f} seconds")

            # Feedback radio
            feedback = st.radio("Was this prediction correct?", ("👍 Yes", "👎 No"))
            if feedback:
                st.write("Thanks for your feedback! We'll use this to improve future versions 😊")

    else:
        st.warning("Please enter a tweet to analyze.")

# Footer
st.markdown("""
# About Creator's
We, Zeeshan & Darab, have developed this project. We are currently in the 8th semester of Computer Science Engineering (CSE).

- **Currently working as**: A Data Engineer @Onix  
- **Contact**: [zeeshanahmed0393@gmail.com](mailto:zeeshanahmed0393@gmail.com)  
- **Feel Free to Reach Out**: [LinkedIn Profile](https://www.linkedin.com/in/zeeshanahmed0393/)
""")
