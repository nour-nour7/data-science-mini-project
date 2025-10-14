# ...existing code...
import streamlit as st
import pandas as pd
from pathlib import Path

# Import the refactored similarity module
import sys
PROJECT_ROOT = Path('/home/selindemirturk/data_science/data-science-mini-project')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from similarity import (
    load_representations,
    load_csvs,
    prepare_book_texts,
    build_tfidf_and_svd,
    recommend_indie_books,
)

# Load book titles from CSV
BOOKS_PATH = PROJECT_ROOT / 'books_clean.csv'
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

# Prepare/load representations once and cache in session_state
if 'reps_loaded' not in st.session_state:
    st.session_state.reps_loaded = False
    st.session_state.X_desc_reduced = None
    st.session_state.X_review_reduced = None
    st.session_state.df_books_text = None
    st.session_state.title_map = None

if not st.session_state.reps_loaded:
    # try loading precomputed representations
    X_desc_reduced, X_review_reduced, assets = load_representations(PROJECT_ROOT / 'models')
    df_text = assets.get('df_books_text')
    title_map = assets.get('title_norm_to_row')
    if X_desc_reduced is None or X_review_reduced is None or df_text is None or title_map is None:
        # build from CSVs (this may take time) — warn the user
        st.info('Building representations from dataset (this may take a while).')
        df_books, df_authors, df_reviews = load_csvs(str(PROJECT_ROOT / 'books_clean.csv'),
                                                    str(PROJECT_ROOT / 'authors_clean.csv'),
                                                    str(PROJECT_ROOT / 'reviews_clean.csv'))
        df_text, title_map = prepare_book_texts(df_books, df_reviews)
        try:
            _, _, X_desc_reduced, _, _, X_review_reduced = build_tfidf_and_svd(df_text, out_dir=PROJECT_ROOT / 'models')
        except Exception as e:
            st.error(f'Failed to build representations: {e}')
            X_desc_reduced = X_review_reduced = None

    st.session_state.X_desc_reduced = X_desc_reduced
    st.session_state.X_review_reduced = X_review_reduced
    st.session_state.df_books_text = df_text
    st.session_state.title_map = title_map
    st.session_state.reps_loaded = True

# Recommendation UI and execution
if len(st.session_state.recognized_books) >= 1:
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    if st.button("Get Recommendations"):
        if st.session_state.X_desc_reduced is None or st.session_state.X_review_reduced is None or st.session_state.df_books_text is None or st.session_state.title_map is None:
            st.error('Representations not ready; cannot compute recommendations.')
        else:
            try:
                recs_df = recommend_indie_books(
                    st.session_state.recognized_books,
                    st.session_state.df_books_text,
                    st.session_state.X_desc_reduced,
                    st.session_state.X_review_reduced,
                    st.session_state.title_map,
                    top_k=10,
                )
                st.session_state.recommendations = recs_df
            except Exception as e:
                st.error(f'Error computing recommendations: {e}')

    if st.session_state.recommendations is not None:
        st.write('Recommendations:')
        recs = st.session_state.recommendations
        # display DataFrame nicely in Streamlit
        st.dataframe(recs)

