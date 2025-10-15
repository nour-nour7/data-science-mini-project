import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import requests
import csv

# Import the similarity module
from similarity import recommend_indie_books

# Run setup on first launch (download data and build models if needed)
# This will only run once when deployed to Streamlit Cloud
try:
    from download_data import setup_project_data
    # Check if data needs to be downloaded
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    if not (PROJECT_ROOT / 'books_clean.csv').exists():
        with st.spinner("First-time setup: Downloading data and building models..."):
            setup_project_data()
except Exception as e:
    st.warning(f"Setup check: {e}. Assuming data is already available.")

# Set up paths - adjust this to your local path
# When running with streamlit, __file__ points to the script location
SCRIPT_DIR = Path(__file__).resolve().parent  # src/
STREAMLIT_DIR = SCRIPT_DIR.parent  # streamlit-hello-world/
PROJECT_ROOT = STREAMLIT_DIR.parent  # data-science-mini-project/

# Models are in data-science-mini-project/models/ (project root)
MODELS_PATH = PROJECT_ROOT / 'models'

# Data files are in data-science-mini-project/ directory (same as PROJECT_ROOT)
BOOKS_PATH = PROJECT_ROOT / 'books_clean.csv'

# Debug output to help troubleshoot paths
st.sidebar.write("### Debug Info")
st.sidebar.caption(f"Models path: {MODELS_PATH}")
st.sidebar.caption(f"Models exists: {MODELS_PATH.exists()}")

# Custom CSS for better styling
st.markdown("""
<style>
    .main * {
        font-size: 18px;
    }
    .stTitle {
        font-size: 36px !important;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for better book search
def search_books(query, books_df, max_results=20):
    """Search books by title or author with substring matching."""
    if not query or len(query) < 2:
        return pd.DataFrame()

    query_lower = query.lower()

    # Search in titles and authors
    title_match = books_df['Title'].str.lower().str.contains(query_lower, na=False, regex=False)
    author_match = books_df['main_author'].str.lower().str.contains(query_lower, na=False, regex=False)

    # Combine matches
    matches = books_df[title_match | author_match].copy()

    # Sort by relevance: exact matches first, then starts-with, then contains
    matches['sort_key'] = 0
    exact_title = matches['Title'].str.lower() == query_lower
    starts_with_title = matches['Title'].str.lower().str.startswith(query_lower)

    matches.loc[exact_title, 'sort_key'] = 3
    matches.loc[starts_with_title & ~exact_title, 'sort_key'] = 2
    matches.loc[~starts_with_title & ~exact_title, 'sort_key'] = 1

    matches = matches.sort_values('sort_key', ascending=False)

    return matches.head(max_results)

# CSV export function
def export_csv(data):
    filename = "indie_recommendations.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'Author', 'Genre', 'Average Rating', 'Is Indie'])
        writer.writerows(data)
    return filename

# Display book with cover image
def display_book(title, author, image_url, avg_rating=None, is_indie=False, genre=None):
    """Display a single book with its cover image and metadata."""
    try:
        if image_url and image_url.strip():
            response = requests.get(image_url, stream=True, timeout=3)
            response.raise_for_status()
            image = Image.open(response.raw)
            resized_image = image.resize((200, 300))
            st.image(resized_image, use_container_width=True)
        else:
            
            st.write("📚 [No Image]")
    except Exception as e:
        st.write("📚 [Image unavailable]")

    # Display book metadata
    st.write(f"**{title}**")
    st.write(f"*by {author}*")
    if genre:
        st.write(f"Genre: {genre}")
    if avg_rating:
        st.write(f"⭐ {avg_rating:.2f}")
    if is_indie:
        st.write("**Indie Author**")

# Main app
st.title("Indie Book Recommender")
st.write("Discover amazing underrated indie authors based on your favorite books!")
st.write("Enter 2-3 books you love, and we'll recommend indie-published books with similar themes and styles.")

# Initialize session state
if 'selected_books' not in st.session_state:
    st.session_state.selected_books = []

# Load book titles for autocomplete
@st.cache_data
def load_book_titles():
    try:
        df = pd.read_csv(BOOKS_PATH)
        return df['Title'].dropna().astype(str).tolist(), df
    except Exception as e:
        st.error(f"Error loading books: {e}")
        st.error(f"Looking for file at: {BOOKS_PATH}")
        return [], pd.DataFrame()

book_titles, books_df = load_book_titles()

if len(book_titles) == 0:
    st.error("Could not load book data. Please ensure books_clean.csv is in the correct location.")
    st.stop()

# Book selection interface
st.subheader("Select Your Favorite Books")
st.write(f"You've selected {len(st.session_state.selected_books)}/3 books")

# Callback to remove book
def remove_book(book_title):
    if book_title in st.session_state.selected_books:
        st.session_state.selected_books.remove(book_title)

# Input field for book search
if len(st.session_state.selected_books) < 3:
    user_input = st.text_input("Search for a book by title or author:",
                                placeholder="e.g., Harry Potter, Stephen King, 1984...",
                                help="Type any part of the book title or author name")

    # Show search results as clickable options
    if user_input and len(user_input) >= 2:
        search_results = search_books(user_input, books_df, max_results=15)

        if len(search_results) > 0:
            st.write(f"**Found {len(search_results)} books:** (click to add)")

            # Display results in a more compact way
            for idx, (_, book) in enumerate(search_results.iterrows()):
                title = book['Title']
                author = book.get('main_author', 'Unknown')
                genre = book.get('genre', '')

                # Skip if already selected
                if title in st.session_state.selected_books:
                    continue

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{title}**")
                    st.caption(f"by {author} • {genre}")
                with col2:
                    if st.button("➕ Add", key=f"add_{idx}_{title[:20]}"):
                        st.session_state.selected_books.append(title)
                        st.rerun()

                if idx < len(search_results) - 1:
                    st.divider()
        else:
            st.info("No books found. Try a different search term.")
    elif user_input and len(user_input) < 2:
        st.info("Type at least 2 characters to search")

# Display selected books
if st.session_state.selected_books:
    st.write("**Your selected books:**")
    cols = st.columns(3)
    for idx, book in enumerate(st.session_state.selected_books):
        with cols[idx % 3]:
            # Get book details
            book_info = books_df[books_df['Title'] == book].iloc[0] if len(books_df[books_df['Title'] == book]) > 0 else None
            if book_info is not None:
                image_url = book_info.get('image', '')
                author = book_info.get('main_author', 'Unknown')
                display_book(book, author, image_url)
            else:
                st.write(f"📚 {book}")

            if st.button(f"❌ Remove", key=f"remove_{idx}"):
                remove_book(book)
                st.rerun()

# Lazy loading - only load when actually needed (on recommendation button click)
@st.cache_resource
def load_matrices_only():
    """Load numpy matrices - the minimum needed for recommendations."""
    import numpy as np
    models_path = Path(MODELS_PATH).resolve()

    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found at {models_path}")

    # Load model matrices
    X_desc = np.load(str(models_path / "X_desc_reduced.npy"), mmap_mode='r')
    X_review = np.load(str(models_path / "X_review_reduced.npy"), mmap_mode='r')
    return X_desc, X_review

@st.cache_resource
def load_title_mapping():
    """Load just the title mapping - small and fast."""
    import json
    models_path = Path(MODELS_PATH).resolve()
    json_file = models_path / "title_norm_to_row.json"

    if not json_file.exists():
        raise FileNotFoundError(f"Title mapping not found at {json_file}. Models path: {models_path}")

    with open(str(json_file), "r") as f:
        return json.load(f)

@st.cache_data
def load_books_metadata():
    """Load minimal book metadata for display - only columns we need."""
    needed_cols = ['Title', 'main_author', 'avg_rating', 'is_indie', 'genre', 'categories', 'image', 'infoLink', 'description']
    df = pd.read_csv(BOOKS_PATH, usecols=needed_cols)
    df['Title_norm'] = df['Title'].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    # Remove duplicates to match the models (which were built on deduplicated data)
    df = df.drop_duplicates(subset=['Title_norm'], keep='first').reset_index(drop=True)
    return df



# Get Recommendations button
if len(st.session_state.selected_books) >= 2:
    st.write("---")
    if st.button("Get Indie Recommendations!", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            # Load models lazily - only when needed
            status.text("Loading recommendation models (step 1/4)...")
            progress.progress(25)
            X_desc_reduced, X_review_reduced = load_matrices_only()

            status.text("Loading book database (step 2/4)...")
            progress.progress(50)
            title_map = load_title_mapping()
            books_metadata = load_books_metadata()

            # Filter books metadata to match model order
            status.text("Preparing data (step 3/4)...")
            progress.progress(65)
            books_in_model = books_metadata[books_metadata['Title_norm'].isin(title_map.keys())].copy()
            books_in_model['row_idx'] = books_in_model['Title_norm'].map(title_map)
            books_in_model = books_in_model.sort_values('row_idx').reset_index(drop=True)
            books_in_model = books_in_model.drop('row_idx', axis=1)

            status.text("Computing similarities (step 4/4)...")
            progress.progress(80)

            # Get recommendations - now returns (indie_df, non_indie_df)
            indie_recs, non_indie_recs = recommend_indie_books(
                st.session_state.selected_books,
                books_in_model,
                X_desc_reduced,
                X_review_reduced,
                title_map,
                top_k=15,
            )

            progress.progress(100)
            status.empty()
            progress.empty()

            st.session_state.indie_recommendations = indie_recs
            st.session_state.non_indie_recommendations = non_indie_recs
            st.success("Found recommendations!")
            st.rerun()

        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f'Error computing recommendations: {e}')
            import traceback
            st.code(traceback.format_exc())

# Display recommendations
if 'indie_recommendations' in st.session_state and 'non_indie_recommendations' in st.session_state:
    indie_recs = st.session_state.indie_recommendations
    non_indie_recs = st.session_state.non_indie_recommendations

    if (indie_recs is not None and len(indie_recs) > 0) or (non_indie_recs is not None and len(non_indie_recs) > 0):
        st.write("---")
        st.header("Your Book Recommendations")
        st.write("Based on your selections, here are books you might enjoy:")

        # Display indie books section
        if indie_recs is not None and len(indie_recs) > 0:
            st.subheader(f"Indie Author Recommendations ({len(indie_recs)} books)")
            st.write("These books are from independent/self-published authors:")

            for idx, (_, book) in enumerate(indie_recs.iterrows()):
                with st.container():
                    col1, col2 = st.columns([1, 4])

                    with col1:
                        # Display book cover
                        image_url = book.get('image', '')
                        try:
                            if image_url and image_url.strip():
                                response = requests.get(image_url, stream=True, timeout=3)
                                response.raise_for_status()
                                image = Image.open(response.raw)
                                resized_image = image.resize((120, 180))
                                st.image(resized_image)
                            else:
                                st.write("📚")
                        except:
                            st.write("📚")

                    with col2:
                        # Display book details
                        title = book.get('Title', 'Unknown')
                        author = book.get('main_author', 'Unknown')
                        avg_rating = book.get('avg_rating', 0)
                        genre = book.get('genre', '')
                        similarity = book.get('similarity', 0)
                        info_link = book.get('infoLink', '')
                        description = book.get('description', '')

                        st.markdown(f"### {title}")
                        st.write(f"**Author:** {author}")
                        if genre:
                            st.write(f"**Genre:** {genre}")
                        if avg_rating and avg_rating > 0:
                            st.write(f"**Rating:** {avg_rating:.2f} / 5.0")
                        st.write(f"**Match Score:** {similarity:.1%}")

                        # Add description in an expander to avoid clutter
                        if description and str(description).strip() and str(description) != 'nan':
                            with st.expander("📖 Read description"):
                                st.write(description)

                        if info_link and info_link.strip():
                            st.markdown(f"[View on Google Books]({info_link})")

                    st.divider()
        else:
            st.info("No indie author books found matching your preferences. Try different books or genres.")

        # Display non-indie books section
        if non_indie_recs is not None and len(non_indie_recs) > 0:
            st.subheader(f"Other Recommendations ({len(non_indie_recs)} books)")
            st.write("These books are from established publishers:")

            for idx, (_, book) in enumerate(non_indie_recs.iterrows()):
                with st.container():
                    col1, col2 = st.columns([1, 4])

                    with col1:
                        # Display book cover
                        image_url = book.get('image', '')
                        try:
                            if image_url and image_url.strip():
                                response = requests.get(image_url, stream=True, timeout=3)
                                response.raise_for_status()
                                image = Image.open(response.raw)
                                resized_image = image.resize((120, 180))
                                st.image(resized_image)
                            else:
                                st.write("📚")
                        except:
                            st.write("📚")

                    with col2:
                        # Display book details
                        title = book.get('Title', 'Unknown')
                        author = book.get('main_author', 'Unknown')
                        avg_rating = book.get('avg_rating', 0)
                        genre = book.get('genre', '')
                        similarity = book.get('similarity', 0)
                        info_link = book.get('infoLink', '')
                        description = book.get('description', '')

                        st.markdown(f"### {title}")
                        st.write(f"**Author:** {author}")
                        if genre:
                            st.write(f"**Genre:** {genre}")
                        if avg_rating and avg_rating > 0:
                            st.write(f"**Rating:** {avg_rating:.2f} / 5.0")
                        st.write(f"**Match Score:** {similarity:.1%}")

                        # Add description in an expander to avoid clutter
                        if description and str(description).strip() and str(description) != 'nan':
                            with st.expander("📖 Read description"):
                                st.write(description)

                        if info_link and info_link.strip():
                            st.markdown(f"[View on Google Books]({info_link})")

                    st.divider()

        # Export to CSV button
        st.write("---")
        all_recs = []
        if indie_recs is not None and len(indie_recs) > 0:
            all_recs.append(indie_recs)
        if non_indie_recs is not None and len(non_indie_recs) > 0:
            all_recs.append(non_indie_recs)

        if all_recs:
            combined_recs = pd.concat(all_recs, ignore_index=True)
            csv_data = [
                (
                    row.get('Title', ''),
                    row.get('main_author', ''),
                    row.get('genre', ''),
                    row.get('avg_rating', 0),
                    row.get('is_indie', False)
                )
                for _, row in combined_recs.iterrows()
            ]

            if st.button("Download All Recommendations as CSV"):
                csv_file = export_csv(csv_data)
                with open(csv_file, 'rb') as f:
                    st.download_button(
                        label="Click to Download",
                        data=f,
                        file_name="book_recs.csv",
                        mime="text/csv"
                    )

            # Show detailed table
            with st.expander("View Detailed Table"):
                display_cols = ['Title', 'main_author', 'genre', 'avg_rating', 'is_indie', 'similarity']
                available_cols = [c for c in display_cols if c in combined_recs.columns]
                st.dataframe(combined_recs[available_cols], use_container_width=True)
    else:
        st.warning("No book recommendations found. Try selecting different books or adjusting your preferences.")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("Our recommender uses machine learning to find indie-published books similar to your favorites.")
    st.write("**How it works:**")
    st.write("1. Select 2-3 books you love")
    st.write("2. Our program analyzes book descriptions and reviews")
    st.write("3. We recommend books from indie authors with similar styles")
    st.write("")
    st.write("**What makes an author 'indie'?**")
    st.write("- Self-published, AND")
    st.write("- Fewer than 3 books published, OR")
    st.write("- Fewer than 20 total reviews")
    st.write("")
    st.caption("Built with ❤️ for underrated writers")
