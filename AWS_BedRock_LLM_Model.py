import streamlit as st
import boto3
import json
import pandas as pd
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Initialize S3 client
s3 = boto3.client("s3")

# Function to load a limited number of rows from the S3 file
@st.cache_data
def load_reviews(bucket, key, limit=2000):
    """
    Load a limited number of rows from the S3 file and return as a DataFrame.
    """
    response = s3.get_object(Bucket=bucket, Key=key)
    csv_data = response['Body'].read().decode('utf-8')
    return pd.read_csv(io.StringIO(csv_data), nrows=limit)

# Function to initialize and cache the Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-runtime")

# Function to invoke the Bedrock model with a custom prompt
def get_model_response(prompt, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"):
    """
    Use the model to analyze the prompt and dataset.
    """
    bedrock_client = get_bedrock_client()
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20000,
            "messages": [{"role": "user", "content": prompt}]
        }),
        accept="application/json",
        contentType="application/json",
    )
    response_body = response["body"].read().decode("utf-8")
    result = json.loads(response_body)
    if "content" in result and len(result["content"]) > 0:
        return result["content"][0].get("text", "No text found in the content.")
    else:
        return "No content found in the response."

# Function to generate dynamic graphs
def generate_graphs(prompt, df):
    """
    Generate graphs dynamically based on the user's prompt.
    """
    try:
        st.subheader("Visualizations")

        # 1. Ratings by Platform
        if "ratings by platform" in prompt.lower():
            st.write("**Ratings by Platform**")
            avg_ratings = df.groupby("Platform")["Ratings"].mean()
            plt.figure(figsize=(10, 6))
            avg_ratings.plot(kind="bar", color="skyblue")
            plt.title("Average Ratings by Platform")
            plt.xlabel("Platform")
            plt.ylabel("Average Ratings")
            plt.tight_layout()
            st.pyplot(plt)

        # 2. Word Cloud for Review Data
        if "word cloud" in prompt.lower():
            st.write("**Word Cloud for Review Data**")
            text = " ".join(str(review) for review in df["Reviews"].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Word Cloud of Reviews")
            plt.tight_layout()
            st.pyplot(plt)

        # 3. Monthly Review Trends
        if "monthly review trends" in prompt.lower():
            st.write("**Monthly Review Trends**")
            monthly_reviews = df.groupby("Month")["Reviews"].sum()
            plt.figure(figsize=(10, 6))
            monthly_reviews.plot(kind="line", marker="o", color="green")
            plt.title("Monthly Review Trends")
            plt.xlabel("Month")
            plt.ylabel("Number of Reviews")
            plt.xticks(range(1, 13))
            plt.tight_layout()
            st.pyplot(plt)

        # 4. Ratings Distribution
        if "ratings distribution" in prompt.lower():
            st.write("**Ratings Distribution**")
            plt.figure(figsize=(10, 6))
            df["Ratings"].value_counts().sort_index().plot(kind="bar", color="orange")
            plt.title("Distribution of Ratings")
            plt.xlabel("Rating")
            plt.ylabel("Count")
            plt.tight_layout()
            st.pyplot(plt)

        # 5. Age Distribution
        if "age distribution" in prompt.lower():
            st.write("**Age Distribution of Reviewers**")
            age_distribution = df["Age"].value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            age_distribution.plot(kind="bar", color="purple")
            plt.title("Age Distribution of Reviewers")
            plt.xlabel("Age")
            plt.ylabel("Count")
            plt.tight_layout()
            st.pyplot(plt)

        # 6. Ratings by Region
        if "ratings by region" in prompt.lower():
            st.write("**Ratings by Region**")
            avg_ratings_region = df.groupby("Region")["Ratings"].mean()
            plt.figure(figsize=(10, 6))
            avg_ratings_region.plot(kind="bar", color="red")
            plt.title("Average Ratings by Region")
            plt.xlabel("Region")
            plt.ylabel("Average Ratings")
            plt.tight_layout()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error generating graphs: {e}")

# Streamlit UI
st.title("Agent - Dark Knight")
st.subheader("Gain valuable insights from customer reviews for Max and D+.")

# S3 bucket and file details
bucket_name = "wbd-hackathon-539247464787-us-west-2-bucket"
file_key = "sentiment_analysis/Dummy_OTT_App_Reviews_Corrected.csv"

# Load and cache the dataset (limited to 2000 rows)
with st.spinner("Loading review data..."):
    reviews_df = load_reviews(bucket_name, file_key, limit=2000)

if st.checkbox("Show raw review data"):
    st.write(reviews_df)

sample_str = "Analyze the customer review data and provide actionable recommendations"
# Input for the dynamic prompt
user_prompt = st.text_area(
    "Enter your query (e.g., 'Ratings by Platform', 'Word Cloud for Review Data', etc.):",
    value=""
)

if st.button("Analyze"):
    with st.spinner("Analyzing the dataset..."):
        # Generate prompt with dataset
        dataset_columns = reviews_df.columns.tolist()
        prompt = f"""
        You are a smart assistant capable of analyzing customer reviews data for OTT platform and generating actionable insights.
        The dataset contains the following columns:
        {', '.join(dataset_columns)}

        Query: {user_prompt}

        Review Data:
        {reviews_df.to_dict(orient="records")}
        """
        # Call the LLM model
        combined_response = get_model_response(prompt)

        # Display the model's analysis
        st.subheader("Model Analysis:")
        st.write(combined_response)

        # Generate graphs based on the prompt
        generate_graphs(user_prompt, reviews_df)
