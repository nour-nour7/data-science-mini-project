import praw
import pandas as pd
from datetime import datetime
import time

# Your Reddit credentials
reddit = praw.Reddit(
    client_id="O1SCYb4rfE9dtd1h5ePYjw",
    client_secret="WPwdwBWUTaFtrJ0SRA9twV37Oos1TA",
    user_agent="bookdatacollection/1.0 by flwrkoo"
)

print(f"Connected to Reddit. Read-only: {reddit.read_only}")

#list of subreddits to scrape
subreddits = [
    'books',
    'wroteabook', 
    'selfpublishing',
    'writing',
    'betareaders',
    'indieauthors'
]

keywords = [
    "recommend", "book", "indie author", "self-published", "hidden gem", "underrated",
    "new release", "debut author", "small press", "indie publisher", "lesser-known",
    "niche book", "discover", "support indie", "support author", "help promote",
    "looking for readers", "reader recommendations", "book club", "reading list",
    "author spotlight","promote", "writing community", "indie fantasy", "indie romance",
    "indie thriller", "indie sci-fi", "indie horror", "indie mystery", "indie historical fiction",
    "indie YA", "announce","release","new book","published"
]

def collect_from_subreddit(subreddit_name, limit=1000):
    print(f"\n=== Collecting from r/{subreddit_name} ===")
    posts = []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            # Filter posts based on keywords (optional)
            if any(keyword.lower() in (post.title + ' ' + (post.selftext if post.selftext else '')).lower() for keyword in keywords):
                posts.append({
                    'title': post.title,
                    'text': post.selftext if post.selftext else 'N/A',
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
                    'author': post.author.name if post.author else 'deleted',
                    'flair': post.link_flair_text if post.link_flair_text else 'N/A',
                    'upvote_ratio': post.upvote_ratio
                })
        
        # Save to CSV
        df = pd.DataFrame(posts)
        filename = f'{subreddit_name}_data.csv'
        df.to_csv(filename, index=False)
        
        print(f"Collected {len(df)} posts from r/{subreddit_name}")
        print(f"Saved to {filename}")
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

all_dataframes = {}

for subreddit_name in subreddits:
    df = collect_from_subreddit(subreddit_name)
    all_dataframes[subreddit_name] = df
    time.sleep(3)


print(f"\n=== FINAL SUMMARY ===")
for subreddit_name, df in all_dataframes.items():
    if not df.empty:
        print(f"r/{subreddit_name}: {len(df)} posts in {subreddit_name}_data.csv")
    else:
        print(f"r/{subreddit_name}: failed to collect")

print(f"\nFiles created:")
for subreddit_name in subreddits:
    print(f"- {subreddit_name}_data.csv")