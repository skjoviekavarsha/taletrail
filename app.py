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
# 📂 Load Dataset
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
# ⚙️ Prepare Recommendation Features
# --------------------------------------------------
df["features"] = df["Genre"] + " " + df["Author"] + " " + df["Type"]
tfidf = TfidfVectorizer(stop_words="english")
feature_matrix = tfidf.fit_transform(df["features"])
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

# --------------------------------------------------
# 📖 Recommendation Functions
# --------------------------------------------------
def recommend_by_book(book_title):
    if book_title not in df["Title"].values:
        return pd.DataFrame()
    idx = df[df["Title"] == book_title].index[0]
    sim_scores = sorted(
        list(enumerate(similarity_matrix[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices][["Title", "Author", "Genre", "Year", "Rating", "Type"]]

def recommend_by_genre(genre):
    subset = df[df["Genre"].str.lower() == genre.lower()]
    return subset.sample(min(5, len(subset)), random_state=42)

# --------------------------------------------------
# 💬 Chatbot Logic
# --------------------------------------------------
def chatbot_reply(query):
    query = query.lower()

    if any(g in query for g in ["hi", "hello", "hey"]):
        return random.choice([
            "👋 Hey there! I’m BookBot — your reading companion!",
            "Hello! Tell me what kind of books you enjoy 📚",
            "Hi reader! Looking for recommendations?"
        ])

    result = df[
        df["Title"].str.lower().str.contains(query) |
        df["Author"].str.lower().str.contains(query) |
        df["Genre"].str.lower().str.contains(query)
    ]

    if not result.empty:
        response = "📚 **Here are some matches:**\n\n"
        for _, r in result.head(3).iterrows():
            response += f"⭐ **{r.Title}** by {r.Author} ({r.Genre}, {r.Year})\n\n"
        return response

    if "recommend" in query or "suggest" in query:
        recs = df.sample(3)
        response = "✨ **You might enjoy:**\n\n"
        for _, r in recs.iterrows():
            response += f"📖 **{r.Title}** by {r.Author} ({r.Genre})\n\n"
        return response

    return "🤖 Try asking: *Recommend fantasy books* or *Books by Colleen Hoover*"

# --------------------------------------------------
# 🧭 MAIN INTERFACE
# --------------------------------------------------
st.title("📚 AI Book Recommendation System")

st.subheader("🔍 Find Similar Books")
book_choice = st.selectbox("Select a Book:", df["Title"].sort_values())
if st.button("Recommend Based on Book"):
    st.dataframe(recommend_by_book(book_choice))

st.subheader("🎨 Find Books by Genre")
genre_choice = st.selectbox("Select a Genre:", sorted(df["Genre"].unique()))
if st.button("Recommend Based on Genre"):
    st.dataframe(recommend_by_genre(genre_choice))

st.subheader("🎯 Filter Books by Rating")
min_rating = st.slider("Minimum Rating", 0.0, 5.0, 4.0)
st.dataframe(df[df["Rating"] >= min_rating])

st.subheader("📖 Popular Books by Genre")
genres = random.sample(list(df["Genre"].unique()), min(4, len(df["Genre"].unique())))
cols = st.columns(len(genres))
for i, genre in enumerate(genres):
    with cols[i]:
        st.markdown(f"**{genre}**")
        subset = df[df["Genre"] == genre].sample(min(3, len(df[df["Genre"] == genre])))
        for _, row in subset.iterrows():
            st.markdown(f"• *{row.Title}* — {row.Author} ⭐{row.Rating}")

# --------------------------------------------------
# 💬 CHAT BUBBLE (SIDEBAR TOGGLE)
# --------------------------------------------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    if st.button("💬 Chat with BookBot"):
        st.session_state.chat_open = not st.session_state.chat_open

# --------------------------------------------------
# 🤖 CHAT WINDOW (FIX 2 APPLIED)
# --------------------------------------------------
if st.session_state.chat_open:
    st.markdown("---")
    st.subheader("🤖 BookBot – Your Reading Assistant")

    for msg in st.session_state.chat_messages:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    user_input = st.text_input("Ask me for book recommendations:")

    if user_input:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_input}
        )

        reply = chatbot_reply(user_input)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": reply}
        )

        st.markdown("### 🤖 BookBot")
        st.markdown(reply)
