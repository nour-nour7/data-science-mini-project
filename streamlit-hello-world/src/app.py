# ...existing code...
import streamlit as st
import pandas as pd

# Load book titles from CSV
BOOKS_PATH = '/home/selindemirturk/data_science/data-science-mini-project/books_clean.csv'
books_df = pd.read_csv(BOOKS_PATH)
book_titles = books_df['Title'].dropna().astype(str).tolist()

def get_suggestions(query, titles, n=5):
    from difflib import get_close_matches
    return get_close_matches(query, titles, n=n, cutoff=0.5)

st.title("Welcome to Indie Author Recommender")
st.write("Type in 2-3 of your favorite books, and get recommendations for indie authors you might enjoy!")
st.write("This app uses the recommend_indie_books function from similarity_score.ipynb (runs the notebook code).")

# Store recognized books in session state
if 'recognized_books' not in st.session_state:
    st.session_state.recognized_books = []

# Callback that will be called by the button
def add_book_callback():
    # read the currently selected suggestion from session_state
    selected = st.session_state.get("book_select", None)
    if selected and selected not in st.session_state.recognized_books:
        st.session_state.recognized_books.append(selected)
    # reset the input field (allowed inside callback)
    st.session_state.book_input = ""

if len(st.session_state.recognized_books) < 3:
    user_input = st.text_input("Type a book name", key="book_input")
    suggestions = get_suggestions(user_input, book_titles) if user_input else []
    if suggestions:
        # create selectbox (its value will be stored in session_state.book_select)
        selected = st.selectbox("Select matching book", suggestions, key="book_select")
        # use callback to append and reset; callback reads session_state.book_select
        st.button("Add Book", key="add_book", on_click=add_book_callback)

st.write("Recognized books:")
for book in st.session_state.recognized_books:
    st.write(f"- {book}")
