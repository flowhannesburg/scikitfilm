import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.linear_model import LinearRegression


class MovieRevenueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Revenue Predictor")
        self.root.geometry("420x280")

        # Data placeholders
        self.df_directors = None
        self.df_movies = None

        # Load Directors CSV
        self.load_dir_btn = tk.Button(root, text="Load Directors CSV", command=self.load_directors_csv)
        self.load_dir_btn.place(x=10, y=10)

        # Load Movies CSV
        self.load_mov_btn = tk.Button(root, text="Load Movies CSV", command=self.load_movies_csv)
        self.load_mov_btn.place(x=200, y=10)

        # Director Name
        tk.Label(root, text="Director's Name").place(x=10, y=60)
        self.dir_input = tk.Entry(root, width=30)
        self.dir_input.place(x=140, y=60)

        # Budget
        tk.Label(root, text="Budget").place(x=10, y=100)
        self.budget_input = tk.Entry(root, width=30)
        self.budget_input.place(x=140, y=100)

        # Predict Button
        self.predict_btn = tk.Button(root, text="Predict Revenue", command=self.predict_revenue)
        self.predict_btn.place(x=10, y=140)

        # Output Label
        self.output_label = tk.Label(root, text="Revenue: ", width=55, anchor="w")
        self.output_label.place(x=10, y=180)

    def load_directors_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Select Directors CSV"
        )
        if file_path:
            try:
                self.df_directors = pd.read_csv(file_path)
                self.output_label.config(text="Directors CSV Loaded Successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Directors CSV: {e}")

    def load_movies_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Select Movies CSV"
        )
        if file_path:
            try:
                self.df_movies = pd.read_csv(file_path)
                self.output_label.config(text="Movies CSV Loaded Successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Movies CSV: {e}")

    def predict_revenue(self):
        # Check both CSVs loaded
        if self.df_directors is None or self.df_movies is None:
            self.output_label.config(text="Please load both CSV files first.")
            return

        director_name = self.dir_input.get().strip()
        budget_str = self.budget_input.get().strip()

        if not budget_str.isdigit():
            self.output_label.config(text="Invalid budget, buddy!")
            return

        budget = float(budget_str)

        # Fuzzy match director name from directors CSV
        all_directors = list(self.df_directors['director_name'].dropna().unique())
        res = process.extractOne(director_name, all_directors, scorer=fuzz.WRatio)

        if not res:
            self.output_label.config(text="No match found.")
            return

        match, score, *_ = res

        if score < 60:
            self.output_label.config(text="No close match for director...")
            return

        # Get director_id from directors CSV
        director_row = self.df_directors[self.df_directors['director_name'] == match]
        if director_row.empty:
            self.output_label.config(text="Director not found in directors CSV.")
            return

        director_id = director_row.iloc[0]['id']

        # Filter movies CSV for this director_id
        sub_df = self.df_movies[self.df_movies['director_id'] == director_id]

        if sub_df.empty:
            self.output_label.config(text="No movies found for this director.")
            return

        # Train simple model: budget → revenue
        if 'budget' not in sub_df.columns or 'revenue' not in sub_df.columns:
            self.output_label.config(text="Movies CSV missing 'budget' or 'revenue' columns.")
            return

        X = sub_df[['budget']].values
        y = sub_df['revenue'].values
        model = LinearRegression()
        model.fit(X, y)

        predicted_revenue = model.predict([[budget]])[0]
        self.output_label.config(
            text=f"Revenue: ${predicted_revenue:,.2f} (Director: {match})"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRevenueApp(root)
    root.mainloop()
