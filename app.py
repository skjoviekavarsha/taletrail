# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# --------------------------------------------------
# 🧩 Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="📚 AI Book Recommendation System", layout="wide")

# --------------------------------------------------
# 📂 Load Dataset with Encoding Fix
# --------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("books.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("books.csv", encoding="ISO-8859-1")

    df = df.fillna("")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    return df

df = load_data()

# --------------------------------------------------
# ⚙️ Prepare Features for Recommendation
# --------------------------------------------------
df["features"] = df["Genre"] + " " + df["Author"] + " " + df["Type"]
tfidf = TfidfVectorizer(stop_words="english")
feature_matrix = tfidf.fit_transform(df["features"])
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

# --------------------------------------------------
# 📖 Book & Genre Recommendation Functions
# --------------------------------------------------
def recommend_by_book(book_title):
    if book_title not in df["Title"].values:
        return pd.DataFrame(columns=["Title", "Author", "Genre", "Rating", "Type"])
    idx = df[df["Title"] == book_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices][["Title", "Author", "Genre", "Year", "Rating", "Type"]]

def recommend_by_genre(genre):
    return df[df["Genre"].str.lower() == genre.lower()].sample(min(5, len(df[df["Genre"].str.lower() == genre.lower()])), random_state=42)

# --------------------------------------------------
# 💬 Chatbot Logic
# --------------------------------------------------
def chatbot_reply(query):
    query = query.lower()

    if any(greet in query for greet in ["hi", "hello", "hey"]):
        return random.choice([
            "👋 Hey there! I’m BookBot — your reading companion!",
            "Hi! Looking for your next great read?",
            "Hello reader! Tell me what kind of books you enjoy 📚"
        ])

    # Search for title/author/genre
    result = df[
        df["Title"].str.lower().str.contains(query) |
        df["Author"].str.lower().str.contains(query) |
        df["Genre"].str.lower().str.contains(query)
    ]

    if not result.empty:
        response = "📚 Here are some matches I found:<br>"
        for _, r in result.head(3).iterrows():
            response += f"⭐ <b>{r.Title}</b> by {r.Author} ({r.Genre}, {r.Year})<br>"
        return response

    elif "recommend" in query or "suggest" in query:
        recs = df.sample(3)
        response = "✨ Here are some recommendations:<br>"
        for _, r in recs.iterrows():
            response += f"📖 <b>{r.Title}</b> by {r.Author} ({r.Genre})<br>"
        return response

    else:
        return "🤖 I can suggest books! Try: *Fantasy books by Sarah J. Maas* or *Recommend romance novels*."


# --------------------------------------------------
# 🧭 Main Interface
# --------------------------------------------------
st.title("📚 AI Book Recommendation System")

# --- Section: Book-based Recommendation ---
st.subheader("🔍 Find Similar Books")
book_choice = st.selectbox("Select a Book:", df["Title"].sort_values())
if st.button("Recommend Based on Book"):
    recommendations = recommend_by_book(book_choice)
    st.dataframe(recommendations)

# --- Section: Genre-based Recommendation ---
st.subheader("🎨 Find Books by Genre")
genre_choice = st.selectbox("Select a Genre:", sorted(df["Genre"].unique()))
if st.button("Recommend Based on Genre"):
    recs = recommend_by_genre(genre_choice)
    st.dataframe(recs)

# --- Section: Filter Books ---
st.subheader("🎯 Filter Books by Rating")
min_rating = st.slider("Minimum Rating", 0.0, 5.0, 4.0)
filtered_df = df[df["Rating"] >= min_rating]
st.dataframe(filtered_df)

# --- Section: Suggested Books by Popular Genres ---
st.subheader("📖 Popular Books by Genre")
genres_to_display = random.sample(list(df["Genre"].unique()), min(4, len(df["Genre"].unique())))
cols = st.columns(len(genres_to_display))
for i, genre in enumerate(genres_to_display):
    with cols[i]:
        st.markdown(f"**{genre}**")
        subset = df[df["Genre"] == genre].sample(min(3, len(df[df["Genre"] == genre])), random_state=42)
        for _, row in subset.iterrows():
            st.markdown(f"• *{row.Title}* — {row.Author} ({row.Year}) ⭐{row.Rating}")
import streamlit.components.v1 as components

# -------------------------
# 🤖 Embed Typebot Widget
# -------------------------
import streamlit.components.v1 as components

typebot_embed = """
<!DOCTYPE html>
<html>
  <head>
    <script src="https://cdn.jsdelivr.net/npm/@typebot.io/embedded"></script>
  </head>
  <body>
    <typebot-bubble
      typebot="my-typebot-02s19hq"
      position="right"
      preview-message="Hi there 👋! Need a book suggestion?"
      color="#1D1D1D"
    ></typebot-bubble>
  </body>
</html>
"""

components.html(typebot_embed, height=600)  # set height so it shows properly

# Handle chatbot query parameter
query_params = st.query_params
if "chat" in query_params:
    q = query_params["chat"]
    st.write(chatbot_reply(q))
