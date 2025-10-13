import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except Exception as e:
    SKLEARN_AVAILABLE = False
    st.error("scikit-learn is not available. Please install it: pip install scikit-learn")


# ------------------------------
# Utilities
# ------------------------------

def _normalize_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def _norm_genre_list(x):
    if isinstance(x, str):
        s = x.lower()
        parts = re.split(r"[;,]", s)
        return {p.strip() for p in parts if p.strip()}
    if isinstance(x, (list, tuple, set)):
        return {str(p).strip().lower() for p in x if str(p).strip()}
    return set()


# ------------------------------
# Data loading and preparation
# ------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_books = pd.read_csv('books_clean.csv')
    df_authors = pd.read_csv('authors_clean.csv')
    df_reviews = pd.read_csv('reviews_clean.csv')
    return df_books, df_authors, df_reviews


def prepare_books_text(df_books: pd.DataFrame, df_reviews: pd.DataFrame) -> pd.DataFrame:
    df_books = df_books.copy()
    df_reviews = df_reviews.copy()

    df_books['Title_norm'] = df_books['Title'].apply(_normalize_title)
    df_reviews['Title_norm'] = df_reviews['Title'].apply(_normalize_title)

    agg_reviews = (
        df_reviews.dropna(subset=['review_text'])
        .groupby('Title_norm', as_index=False)
        .agg({
            'review_text': lambda s: " \n".join(map(str, s)),
            'rating': 'mean',
        })
        .rename(columns={'rating': 'avg_review_rating_from_reviews'})
    )

    df_books_text = df_books.merge(agg_reviews, on='Title_norm', how='left')
    df_books_text['review_text_agg'] = df_books_text['review_text'].fillna("")
    df_books_text['desc_text'] = df_books_text['description'].fillna("")

    mask_sparse_reviews = df_books_text['review_text_agg'].str.len() < 50
    df_books_text.loc[mask_sparse_reviews, 'review_text_agg'] = (
        df_books_text.loc[mask_sparse_reviews, 'review_text_agg'] +
        " \n" + df_books_text.loc[mask_sparse_reviews, 'desc_text'].astype(str)
    )

    has_any_text = (
        df_books_text['desc_text'].str.strip().str.len() > 0
    ) | (
        df_books_text['review_text_agg'].str.strip().str.len() > 0
    )
    df_books_text = df_books_text[has_any_text].reset_index(drop=True)

    return df_books_text


# ------------------------------
# Vectorizers and SVD models (cached)
# ------------------------------

@st.cache_resource(show_spinner=False)
def build_models(
    df_books_text: pd.DataFrame,
    desc_max_features: int = 20000,
    rev_max_features: int = 12000,
    min_df: int = 10,
    svd_components: int = 200,
):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required to build models.")

    tfidf_desc = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.8,
        min_df=min_df,
        ngram_range=(1, 1),
        max_features=desc_max_features,
        dtype=np.float32,
    )
    X_desc = tfidf_desc.fit_transform(df_books_text['desc_text'])

    tfidf_rev = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.8,
        min_df=min_df,
        ngram_range=(1, 1),
        max_features=rev_max_features,
        dtype=np.float32,
    )
    X_rev = tfidf_rev.fit_transform(df_books_text['review_text_agg'])

    # SVD reductions
    svd_desc_comp = max(2, min(svd_components, X_desc.shape[1] - 1))
    svd_rev_comp = max(2, min(svd_components, X_rev.shape[1] - 1))

    svd_desc = TruncatedSVD(n_components=svd_desc_comp, random_state=42)
    X_desc_reduced = svd_desc.fit_transform(X_desc)
    X_desc_reduced = X_desc_reduced / (np.linalg.norm(X_desc_reduced, axis=1, keepdims=True) + 1e-12)

    svd_rev = TruncatedSVD(n_components=svd_rev_comp, random_state=42)
    X_review_reduced = svd_rev.fit_transform(X_rev)
    X_review_reduced = X_review_reduced / (np.linalg.norm(X_review_reduced, axis=1, keepdims=True) + 1e-12)

    title_norm_to_row = {t: i for i, t in enumerate(df_books_text['Title_norm'])}
    return {
        'tfidf_desc': tfidf_desc,
        'tfidf_rev': tfidf_rev,
        'X_desc_reduced': X_desc_reduced,
        'X_review_reduced': X_review_reduced,
        'title_norm_to_row': title_norm_to_row,
    }


# ------------------------------
# Recommendation logic
# ------------------------------

def recommend_indie_books(
    favorite_titles: List[str],
    df_books_text: pd.DataFrame,
    model_bundle: dict,
    top_k: int = 10,
    genre_weight: float = 0.6,
    review_weight: float = 0.4,
    indie_boost: float = 0.1,
    exclude_favorites: bool = True,
) -> pd.DataFrame:
    X_desc_reduced = model_bundle['X_desc_reduced']
    X_review_reduced = model_bundle['X_review_reduced']
    title_norm_to_row = model_bundle['title_norm_to_row']

    norm_favs = [_normalize_title(t) for t in favorite_titles]
    indices = [title_norm_to_row.get(t) for t in norm_favs]
    fav_idx = [i for i in indices if i is not None]
    if not fav_idx:
        return pd.DataFrame(columns=['Title', 'similarity'])

    pref_desc = X_desc_reduced[fav_idx].mean(axis=0, keepdims=True)
    pref_desc = pref_desc / (np.linalg.norm(pref_desc, axis=1, keepdims=True) + 1e-12)

    pref_rev = X_review_reduced[fav_idx].mean(axis=0, keepdims=True)
    pref_rev = pref_rev / (np.linalg.norm(pref_rev, axis=1, keepdims=True) + 1e-12)

    sims_desc = (pref_desc @ X_desc_reduced.T).ravel()
    sims_rev = (pref_rev @ X_review_reduced.T).ravel()

    fav_genres_sets = []
    if 'categories' in df_books_text.columns:
        for i in fav_idx:
            fav_genres_sets.append(_norm_genre_list(df_books_text.loc[i, 'categories']))
    fav_genres_union = set().union(*fav_genres_sets) if fav_genres_sets else set()

    book_genres = df_books_text['categories'].fillna("").apply(_norm_genre_list) if 'categories' in df_books_text.columns else pd.Series([set()] * len(df_books_text))
    genre_match_mask = book_genres.apply(lambda g: len(g.intersection(fav_genres_union)) > 0) if fav_genres_union else pd.Series([True] * len(df_books_text))

    combined = genre_weight * sims_desc + review_weight * sims_rev
    indie_mask = (df_books_text['is_indie'] == True) if 'is_indie' in df_books_text.columns else pd.Series([False] * len(df_books_text))
    combined = combined + indie_boost * indie_mask.values.astype(float)

    result = df_books_text.copy()
    result['similarity'] = combined

    aligned_mask = genre_match_mask.reindex(result.index, fill_value=False)
    if aligned_mask.any():
        result = result[aligned_mask.to_numpy()]
    # Else: fall back to overall combined score (no filtering)

    if exclude_favorites:
        result = result[~result['Title_norm'].isin(norm_favs)]

    cols = ['Title','main_author','avg_rating','is_indie','genre','categories','similarity','previewLink','infoLink']
    existing_cols = [c for c in cols if c in result.columns]
    return result.sort_values(by='similarity', ascending=False)[existing_cols].head(top_k)


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Indie Book Recommender", layout="wide")
st.title("📚 Indie Book Recommender")
st.caption("Type 2–3 favorite books to get indie recommendations with genre-aware, description-focused similarity.")

with st.spinner("Loading data…"):
    df_books, df_authors, df_reviews = load_data()
    df_books_text = prepare_books_text(df_books, df_reviews)

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    genre_weight = st.slider("Genre/Description weight", 0.0, 1.0, 0.6, 0.05)
    review_weight = st.slider("Reviews weight", 0.0, 1.0, 0.4, 0.05)
    total = max(genre_weight + review_weight, 1e-9)
    genre_weight, review_weight = genre_weight/total, review_weight/total

    indie_boost = st.slider("Indie boost", 0.0, 0.5, 0.1, 0.01)
    svd_components = st.slider("SVD components", 50, 400, 200, 10)
    desc_max_features = st.select_slider("Desc TF-IDF max_features", options=[5000, 10000, 20000, 30000, 50000], value=20000)
    rev_max_features = st.select_slider("Review TF-IDF max_features", options=[3000, 6000, 10000, 12000, 20000, 30000], value=12000)
    min_df = st.select_slider("min_df", options=[2,5,10,15,20], value=10)

# Build models (cached)
with st.spinner("Building models… (cached)"):
    model_bundle = build_models(
        df_books_text,
        desc_max_features=desc_max_features,
        rev_max_features=rev_max_features,
        min_df=min_df,
        svd_components=svd_components,
    )

# Search + selection UI
st.subheader("Pick 2–3 favorite books")
search = st.text_input("Search titles", "")
all_titles = df_books_text['Title'].tolist()
if search.strip():
    filt = search.strip().lower()
    options = [t for t in all_titles if filt in t.lower()]
    options = options[:500]  # keep UI snappy
else:
    options = all_titles[:1000]  # limit initial options

selected = st.multiselect("Favorites", options=options, max_selections=3)

col1, col2 = st.columns([1, 2])
with col1:
    run = st.button("Recommend")

if run:
    if len(selected) < 1:
        st.warning("Please select at least 1 favorite (ideally 2–3).")
    else:
        with st.spinner("Scoring recommendations…"):
            out = recommend_indie_books(
                favorite_titles=selected,
                df_books_text=df_books_text,
                model_bundle=model_bundle,
                top_k=20,
                genre_weight=genre_weight,
                review_weight=review_weight,
                indie_boost=indie_boost,
                exclude_favorites=True,
            )
        if out.empty:
            st.info("No results found. Try different favorites or relax settings.")
        else:
            st.success(f"Found {len(out)} recommendations")
            # Optional: create a link column if preview/info links exist
            if 'previewLink' in out.columns or 'infoLink' in out.columns:
                link_col = []
                for _, row in out.iterrows():
                    links = []
                    if 'previewLink' in out.columns and pd.notna(row.get('previewLink', None)):
                        links.append(f"[Preview]({row['previewLink']})")
                    if 'infoLink' in out.columns and pd.notna(row.get('infoLink', None)):
                        links.append(f"[Info]({row['infoLink']})")
                    link_col.append(" | ".join(links))
                out = out.copy()
                out.insert(len(out.columns), 'Links', link_col)

            st.dataframe(out.reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.caption("Tip: Use the sidebar to adjust weights and SVD/TF‑IDF sizes if you run into memory constraints.")
