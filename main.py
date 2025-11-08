import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from tkinter import *
from tkinter import messagebox, scrolledtext

# ===============================
# 📊 Load Data
# ===============================
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
links = pd.read_csv("links.csv")

# Merge IMDb links
movies = movies.merge(links, on="movieId", how="left")
movies["imdb_url"] = "https://www.imdb.com/title/tt" + movies["imdbId"].astype(str).str.zfill(7) + "/"

# ===============================
# 🤖 Train Model
# ===============================
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
algo = SVD()
algo.fit(trainset)

# ===============================
# 🎯 Recommendation Function
# ===============================
def get_top_n_recommendations(user_id, n=10):
    # All movie IDs
    all_movie_ids = movies["movieId"].unique()
    rated_movies = ratings[ratings["userId"] == user_id]["movieId"].unique()
    unseen_movies = [m for m in all_movie_ids if m not in rated_movies]

    predictions = [algo.predict(user_id, mid) for mid in unseen_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]

    results = []
    for pred in top_n:
        movie_info = movies[movies["movieId"] == pred.iid].iloc[0]
        results.append({
            "title": movie_info["title"],
            "score": round(pred.est, 2),
            "imdb": movie_info["imdb_url"]
        })
    return results

# ===============================
# 🪟 Tkinter GUI
# ===============================
root = Tk()
root.title("🎬 Movie Recommendation System")
root.geometry("720x600")
root.configure(bg="#121212")

# Title Label
Label(root, text="Movie Recommendation System", font=("Helvetica", 20, "bold"), fg="#00BFFF", bg="#121212").pack(pady=20)

# User Input Frame
frame = Frame(root, bg="#121212")
frame.pack(pady=10)
Label(frame, text="Enter User ID:", font=("Helvetica", 12), fg="white", bg="#121212").grid(row=0, column=0, padx=5)
user_entry = Entry(frame, width=10, font=("Helvetica", 12))
user_entry.grid(row=0, column=1, padx=5)

# Output Box (Scrollable)
output_box = scrolledtext.ScrolledText(root, width=80, height=20, font=("Consolas", 11), bg="#1E1E1E", fg="white", wrap=WORD)
output_box.pack(padx=15, pady=15)

# ===============================
# ⚡ Action Function
# ===============================
def show_recommendations():
    output_box.delete(1.0, END)
    try:
        user_id = int(user_entry.get())
        if user_id not in ratings["userId"].unique():
            messagebox.showerror("Error", f"User ID {user_id} not found in dataset!")
            return

        recs = get_top_n_recommendations(user_id, 10)
        output_box.insert(END, f"🎯 Top 10 Recommendations for User {user_id}:\n\n")

        for i, rec in enumerate(recs, 1):
            output_box.insert(END, f"{i}. {rec['title']} ⭐ {rec['score']}\n")
            output_box.insert(END, f"   IMDb: {rec['imdb']}\n\n")

    except ValueError:
        messagebox.showwarning("Invalid Input", "Please enter a valid numeric User ID!")

# Button
Button(root, text="Get Recommendations", command=show_recommendations,
       bg="#00BFFF", fg="white", font=("Helvetica", 12, "bold"),
       activebackground="#0080FF", relief="flat", padx=15, pady=8, cursor="hand2").pack(pady=10)

root.mainloop()
