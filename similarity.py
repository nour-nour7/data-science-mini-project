import pandas as pd
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except Exception as e:
    SKLEARN_AVAILABLE = False
    print("scikit-learn is not available. Please install it to run the recommender.")
    print("Error:", e)

import re
from typing import List, Tuple


# ---------- Helpers ----------
def _normalize_title(s: str) -> str:
    """Normalize titles for robust matching."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def prepare_books_text(df_books: pd.DataFrame, df_reviews: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Aggregate reviews per book, merge with books, and create the text corpus.

    Returns:
        df_books_text: Books with 'text_corpus' column and metadata
        title_norm_to_row: mapping for fast lookup
    """
    df_books['Title_norm'] = df_books['Title'].apply(_normalize_title)
    df_reviews['Title_norm'] = df_reviews['Title'].apply(_normalize_title)

    # Aggregate review texts by Title
    agg_reviews = (
        df_reviews
        .dropna(subset=['review_text'])
        .groupby('Title_norm', as_index=False)
        .agg({
            'review_text': lambda s: " \n".join(map(str, s)),
            'rating': 'mean'
        })
        .rename(columns={'rating': 'avg_review_rating_from_reviews'})
    )

    # Merge with books; also create a fallback text using description when no reviews
    df_books_text = df_books.merge(agg_reviews, on='Title_norm', how='left')
    df_books_text['text_corpus'] = df_books_text['review_text'].fillna("")

    # Optional: enrich text with description if reviews are sparse
    desc_fill = df_books_text['description'].fillna("")
    mask_sparse = df_books_text['text_corpus'].str.len() < 50
    df_books_text.loc[mask_sparse, 'text_corpus'] = (
        df_books_text.loc[mask_sparse, 'text_corpus'] + " \n" + desc_fill[mask_sparse].astype(str)
    )

    # Keep only rows with some text
    df_books_text = df_books_text[df_books_text['text_corpus'].str.strip().str.len() > 0].reset_index(drop=True)

    title_norm_to_row = {t: i for i, t in enumerate(df_books_text['Title_norm'])}
    return df_books_text, title_norm_to_row


def build_tfidf(
    df_books_text: pd.DataFrame,
    *,
    max_features: int = 20000,
    min_df: int = 10,
    ngram_range: tuple = (1, 1),
):
    """Build a memory-friendly TF-IDF matrix from the text corpus."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required to build TF-IDF. Please install scikit-learn.")
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.8,
        min_df=min_df,
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
    )
    X = tfidf.fit_transform(df_books_text['text_corpus'])
    return tfidf, X


def reduce_with_svd(X, n_components: int = 300):
    """Optional TruncatedSVD reduction; returns L2-normalized reduced matrix.
    If it cannot run, returns (False, None) to signal fallback.
    """
    try:
        if X.shape[1] <= 1 or n_components <= 1:
            return False, None
        n_comp = int(min(n_components, X.shape[1] - 1))
        if n_comp <= 0:
            return False, None
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_reduced = svd.fit_transform(X)
        # L2-normalize rows so cosine ~ dot
        norms = np.linalg.norm(X_reduced, axis=1, keepdims=True) + 1e-12
        X_reduced = X_reduced / norms
        return True, X_reduced
    except Exception as e:
        print("SVD step skipped due to error:", e)
        return False, None


def recommend_indie_books(
    favorite_titles: List[str],
    df_books_text: pd.DataFrame,
    title_norm_to_row: dict,
    X,
    use_svd: bool = False,
    X_reduced=None,
    top_k: int = 10,
    exclude_favorites: bool = True,
) -> pd.DataFrame:
    """Recommend indie books using cosine similarity on TF-IDF (or SVD) space."""
    if X.shape[0] == 0:
        return pd.DataFrame(columns=['Title', 'similarity'])

    norm_favs = [_normalize_title(t) for t in favorite_titles]
    indices = [title_norm_to_row.get(t) for t in norm_favs]
    found = [i for i in indices if i is not None]
    if not found:
        print("No favorite titles found in the corpus. Check spelling or availability.")
        return pd.DataFrame(columns=['Title', 'similarity'])

    if use_svd and X_reduced is not None:
        pref_vec = X_reduced[found].mean(axis=0, keepdims=True)
        sims = (pref_vec @ X_reduced.T).ravel()
    else:
        pref_vec = X[found].mean(axis=0)
        sims = cosine_similarity(pref_vec, X).ravel()

    result = df_books_text.copy()
    result['similarity'] = sims
    if exclude_favorites:
        result = result[~result['Title_norm'].isin(norm_favs)]
    result = result[result['is_indie'] == True]

    cols = ['Title', 'main_author', 'avg_rating', 'is_indie', 'similarity', 'previewLink', 'infoLink', 'categories']
    existing_cols = [c for c in cols if c in result.columns]
    return result.sort_values(by='similarity', ascending=False)[existing_cols].head(top_k)


def build_corpus_subset(
    df_books_text: pd.DataFrame,
    favorite_titles: List[str],
    mode: str = "indie_plus_favorites",
):
    """Build a smaller corpus for TF-IDF to reduce memory usage.

    mode="indie_plus_favorites": keep all indie books + any favorites (even if not indie).
    mode="all_books": keep all books (original behavior).

    Returns: (df_subset, title_norm_to_row_subset)
    """
    if mode == "all_books":
        subset = df_books_text.copy().reset_index(drop=True)
        mapping = {t: i for i, t in enumerate(subset['Title_norm'])}
        return subset, mapping

    norm_favs = {_normalize_title(t) for t in favorite_titles}
    mask = (df_books_text['is_indie'] == True) | (df_books_text['Title_norm'].isin(norm_favs))
    subset = df_books_text[mask].reset_index(drop=True)
    mapping = {t: i for i, t in enumerate(subset['Title_norm'])}
    return subset, mapping

def main():
    # Config (edit here, no CLI)
    USE_SVD = True
    SVD_COMPONENTS = 200  # lower components for memory
    CORPUS_MODE = "indie_plus_favorites"  # or "all_books"
    TFIDF_MAX_FEATURES = 20000
    TFIDF_MIN_DF = 10
    TFIDF_NGRAM_RANGE = (1, 1)

    # Load data
    df_books = pd.read_csv('books_clean.csv')
    df_authors = pd.read_csv('authors_clean.csv')  # not used directly but kept for completeness
    df_reviews = pd.read_csv('reviews_clean.csv')

    # Prepare corpus
    df_books_text, _ = prepare_books_text(df_books, df_reviews)
    is_indie_count = int((df_books_text['is_indie'] == True).sum())
    print(f"Prepared text for {len(df_books_text)} books; indie count in this set: {is_indie_count}")

    if not SKLEARN_AVAILABLE:
        print("Please install scikit-learn to continue.")
        return

    # Choose favorites (edit titles here)
    example_favorites: List[str] = []
    if len(df_books_text) >= 1:
        example_favorites.append(df_books_text['Title'].iloc[0])
    if len(df_books_text) >= 2:
        example_favorites.append(df_books_text['Title'].iloc[1])

    if not example_favorites:
        print("No books available to generate recommendations.")
        return

    # Build a smaller corpus (indie + favorites) to reduce memory
    df_subset, title_norm_to_row_subset = build_corpus_subset(
        df_books_text,
        favorite_titles=example_favorites,
        mode=CORPUS_MODE,
    )
    subset_indie_count = int((df_subset['is_indie'] == True).sum())
    print(f"Corpus subset size: {len(df_subset)} (indie: {subset_indie_count})")

    # TF-IDF on subset
    _, X = build_tfidf(
        df_subset,
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    print(f"TF-IDF matrix shape (subset): {X.shape}")

    # Optional SVD on subset
    X_reduced = None
    use_svd_effective = False
    if USE_SVD:
        use_svd_effective, X_reduced = reduce_with_svd(X, n_components=SVD_COMPONENTS)
        if use_svd_effective:
            print(f"SVD reduced matrix shape: {X_reduced.shape}")
        else:
            print("SVD was not applied (insufficient features or error). Falling back to TF-IDF space.")

    # Run recommendations
    print("Favorites used for demo:", example_favorites)
    try:
        recs = recommend_indie_books(
            favorite_titles=example_favorites,
            df_books_text=df_subset,
            title_norm_to_row=title_norm_to_row_subset,
            X=X,
            use_svd=use_svd_effective,
            X_reduced=X_reduced,
            top_k=10,
            exclude_favorites=True,
        )
        if recs.empty:
            print("No indie recommendations found. Try different favorites or adjust parameters.")
        else:
            print("Top indie recommendations:")
            print(recs.to_string(index=False))
    except Exception as e:
        print("Recommendation failed:", e)


if __name__ == "__main__":
    main()