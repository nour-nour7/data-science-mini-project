import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import joblib

# Optional sklearn imports; checked at runtime
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Default artifacts directory - use project root models/
# Navigate up from wherever this script is run to the data-science-mini-project root
SCRIPT_DIR = Path(__file__).resolve().parent  # src/
STREAMLIT_DIR = SCRIPT_DIR.parent  # streamlit-hello-world/
PROJECT_ROOT = STREAMLIT_DIR.parent  # data-science-mini-project/
OUT_DIR = PROJECT_ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)


def load_csvs(books_path: str = "books_clean.csv",
              authors_path: str = "authors_clean.csv",
              reviews_path: str = "reviews_clean.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSV files and return dataframes."""
    df_books = pd.read_csv(books_path)
    df_authors = pd.read_csv(authors_path)
    df_reviews = pd.read_csv(reviews_path)
    return df_books, df_authors, df_reviews


def _normalize_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def prepare_book_texts(df_books: pd.DataFrame, df_reviews: pd.DataFrame,
                       desc_col: str = "description",
                       review_col: str = "review_text") -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Aggregate reviews, normalize titles and prepare text columns for modeling.

    Returns (df_books_text, title_norm_to_row)
    """
    df = df_books.copy()
    df["Title_norm"] = df["Title"].apply(_normalize_title)

    reviews = df_reviews.copy()
    reviews["Title_norm"] = reviews["Title"].apply(_normalize_title)

    agg_reviews = (
        reviews
        .dropna(subset=[review_col])
        .groupby("Title_norm", as_index=False)
        .agg({review_col: lambda s: " \n".join(map(str, s)), "rating": "mean"})
        .rename(columns={"rating": "avg_review_rating_from_reviews", review_col: "review_text"})
    )

    df_text = df.merge(agg_reviews, on="Title_norm", how="left")
    df_text["review_text_agg"] = df_text["review_text"].fillna("")
    df_text["desc_text"] = df_text.get(desc_col, pd.Series("", index=df_text.index)).fillna("")

    # enrich sparse reviews with description
    mask_sparse = df_text["review_text_agg"].str.len() < 50
    df_text.loc[mask_sparse, "review_text_agg"] = (
        df_text.loc[mask_sparse, "review_text_agg"] + " \n" + df_text.loc[mask_sparse, "desc_text"].astype(str)
    )

    has_any_text = (df_text["desc_text"].str.strip().str.len() > 0) | (df_text["review_text_agg"].str.strip().str.len() > 0)
    df_text = df_text[has_any_text].reset_index(drop=True)

    # Remove duplicate titles - keep first occurrence
    df_text = df_text.drop_duplicates(subset=["Title_norm"], keep="first").reset_index(drop=True)

    title_norm_to_row = {t: i for i, t in enumerate(df_text["Title_norm"])}
    return df_text, title_norm_to_row


def build_tfidf_and_svd(df_books_text: pd.DataFrame,
                        desc_col: str = "desc_text",
                        rev_col: str = "review_text_agg",
                        out_dir: Path = OUT_DIR,
                        svd_components: int = 300,
                        min_df_desc: int = 5,
                        min_df_rev: int = 5):
    """Fit TF-IDF on description and reviews, reduce with SVD, save artifacts and return them.

    Returns (tfidf_desc, svd_desc, X_desc_reduced, tfidf_rev, svd_rev, X_rev_reduced)
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required to build TF-IDF/SVD. Please install it.")

    tfidf_desc = TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.8, min_df=min_df_desc,
                                 ngram_range=(1, 1), max_features=50000, dtype=np.float32)
    X_desc = tfidf_desc.fit_transform(df_books_text[desc_col])

    tfidf_rev = TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.8, min_df=min_df_rev,
                                ngram_range=(1, 1), max_features=30000, dtype=np.float32)
    X_rev = tfidf_rev.fit_transform(df_books_text[rev_col])

    desc_comp = max(2, min(svd_components, X_desc.shape[1] - 1))
    rev_comp = max(2, min(svd_components, X_rev.shape[1] - 1))

    svd_desc = TruncatedSVD(n_components=desc_comp, random_state=42)
    X_desc_reduced = svd_desc.fit_transform(X_desc)
    X_desc_reduced = X_desc_reduced / (np.linalg.norm(X_desc_reduced, axis=1, keepdims=True) + 1e-12)

    svd_rev = TruncatedSVD(n_components=rev_comp, random_state=42)
    X_rev_reduced = svd_rev.fit_transform(X_rev)
    X_rev_reduced = X_rev_reduced / (np.linalg.norm(X_rev_reduced, axis=1, keepdims=True) + 1e-12)

    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "X_desc_reduced.npy", X_desc_reduced)
    np.save(out_dir / "X_review_reduced.npy", X_rev_reduced)
    joblib.dump(tfidf_desc, out_dir / "tfidf_desc.joblib")
    joblib.dump(svd_desc, out_dir / "svd_desc.joblib")
    joblib.dump(tfidf_rev, out_dir / "tfidf_rev.joblib")
    joblib.dump(svd_rev, out_dir / "svd_rev.joblib")
    df_books_text.to_parquet(out_dir / "df_books_text.parquet", index=False)
    # Save title mapping - MUST match the exact row order of the matrices and dataframe
    title_norm_to_row = {t: i for i, t in enumerate(df_books_text["Title_norm"])}
    with open(out_dir / "title_norm_to_row.json", "w") as f:
        json.dump({k: int(v) for k, v in title_norm_to_row.items()}, f)

    return tfidf_desc, svd_desc, X_desc_reduced, tfidf_rev, svd_rev, X_rev_reduced


def load_representations(out_dir: Path = OUT_DIR) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Load reduced matrices and optional assets if present.

    Returns (X_desc_reduced, X_review_reduced, assets_dict)
    """
    assets: Dict[str, object] = {}
    npz_path = out_dir / "reduced_matrices.npz"
    desc_npy = out_dir / "X_desc_reduced.npy"
    rev_npy = out_dir / "X_review_reduced.npy"

    X_desc_reduced = X_review_reduced = None
    if npz_path.exists():
        with np.load(npz_path) as data:
            X_desc_reduced = data.get("X_desc")
            X_review_reduced = data.get("X_review")
    elif desc_npy.exists() and rev_npy.exists():
        X_desc_reduced = np.load(desc_npy)
        X_review_reduced = np.load(rev_npy)

    for name in ("tfidf_desc.joblib", "svd_desc.joblib", "tfidf_rev.joblib", "svd_rev.joblib"):
        p = out_dir / name
        if p.exists():
            try:
                assets[name.split(".")[0]] = joblib.load(p)
            except Exception:
                assets[name.split(".")[0]] = None

    title_map_p = out_dir / "title_norm_to_row.json"
    if title_map_p.exists():
        with open(title_map_p, "r") as f:
            assets["title_norm_to_row"] = json.load(f)

    df_p = out_dir / "df_books_text.parquet"
    if df_p.exists():
        try:
            assets["df_books_text"] = pd.read_parquet(df_p)
        except Exception:
            assets["df_books_text"] = None

    return X_desc_reduced, X_review_reduced, assets


# Recommendation defaults
GENRE_WEIGHT = 0.6
REVIEW_WEIGHT = 0.4
INDIE_BOOST = 0.1


def _norm_genre_list(x) -> set:
    if isinstance(x, str):
        s = x.lower()
        parts = re.split(r"[;,]", s)
        return {p.strip() for p in parts if p.strip()}
    if isinstance(x, (list, tuple, set)):
        return {str(p).strip().lower() for p in x if str(p).strip()}
    return set()


def recommend_indie_books(favorite_titles: List[str],
                          df_books_text: pd.DataFrame,
                          X_desc_reduced: np.ndarray,
                          X_review_reduced: np.ndarray,
                          title_norm_to_row: Dict[str, int],
                          top_k: int = 10,
                          exclude_favorites: bool = True,
                          genre_weight: float = GENRE_WEIGHT,
                          review_weight: float = REVIEW_WEIGHT,
                          indie_boost: float = INDIE_BOOST) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recommend books prioritizing genre/description similarity, with indie boost.

    Returns:
        Tuple of (indie_books_df, non_indie_books_df) - both sorted by similarity
    """
    if X_desc_reduced is None or X_review_reduced is None:
        raise RuntimeError("Reduced representations required. Build or load them first.")

    norm_favs = [_normalize_title(t) for t in favorite_titles]
    indices = [title_norm_to_row.get(t) for t in norm_favs]
    fav_idx = [i for i in indices if i is not None]
    if not fav_idx:
        empty_cols = ['Title', 'similarity']
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    pref_desc = X_desc_reduced[fav_idx].mean(axis=0, keepdims=True)
    pref_desc = pref_desc / (np.linalg.norm(pref_desc, axis=1, keepdims=True) + 1e-12)
    pref_rev = X_review_reduced[fav_idx].mean(axis=0, keepdims=True)
    pref_rev = pref_rev / (np.linalg.norm(pref_rev, axis=1, keepdims=True) + 1e-12)

    sims_desc = (pref_desc @ X_desc_reduced.T).ravel()
    sims_rev = (pref_rev @ X_review_reduced.T).ravel()

    fav_genres_sets = [_norm_genre_list(df_books_text.loc[i, 'categories'] if 'categories' in df_books_text.columns else '') for i in fav_idx]
    fav_genres_union = set().union(*fav_genres_sets) if fav_genres_sets else set()
    book_genres = df_books_text['categories'].fillna("").apply(_norm_genre_list)
    genre_match_mask = book_genres.apply(lambda g: len(g.intersection(fav_genres_union)) > 0)

    # Calculate similarity - boost books that match genre
    combined = genre_weight * sims_desc + review_weight * sims_rev

    # Add genre match bonus instead of filtering (0.15 boost for matching genre)
    genre_match_bonus = genre_match_mask.astype(float) * 0.15
    combined = combined + genre_match_bonus

    result = df_books_text.copy()
    result['similarity'] = combined

    # Don't filter by genre - let similarity scoring handle it
    # This way we don't exclude potentially good matches with missing/poor genre tags

    if exclude_favorites:
        norm_favs_set = set(norm_favs)
        result = result[~result['Title_norm'].isin(norm_favs_set)]

    cols = ['Title', 'main_author', 'avg_rating', 'is_indie', 'genre', 'categories', 'similarity', 'previewLink', 'infoLink', 'image']
    existing_cols = [c for c in cols if c in result.columns]
    result = result[existing_cols]

    # Split into indie and non-indie
    indie_mask = (result.get('is_indie') == True)
    indie_books = result[indie_mask].sort_values(by='similarity', ascending=False).head(top_k)
    non_indie_books = result[~indie_mask].sort_values(by='similarity', ascending=False).head(top_k)

    return indie_books, non_indie_books


if __name__ == '__main__':
    # quick demo
    try:
        df_books, df_authors, df_reviews = load_csvs()
        df_books_text, title_map = prepare_book_texts(df_books, df_reviews)
        X_desc_reduced, X_review_reduced, assets = load_representations()
        if X_desc_reduced is None or X_review_reduced is None:
            if not SKLEARN_AVAILABLE:
                raise RuntimeError('scikit-learn not available to build representations.')
            _, _, X_desc_reduced, _, _, X_review_reduced = build_tfidf_and_svd(df_books_text)

        example_fav = [df_books_text['Title'].iloc[0]] if len(df_books_text) > 0 else []
        indie_recs, non_indie_recs = recommend_indie_books(example_fav, df_books_text, X_desc_reduced, X_review_reduced, title_map, top_k=10)
        print('Example favorites:', example_fav)
        print(f'Indie recommendations: {len(indie_recs)}')
        print(indie_recs.head())
        print(f'Non-indie recommendations: {len(non_indie_recs)}')
        print(non_indie_recs.head())
    except Exception as e:
        print('Demo failed:', e)