import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
import requests
from io import BytesIO
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# TMDB API configuration
api_key = "0625ea93464ddd22fbc2048981f3acf6"
base_url = "https://api.themoviedb.org/3"

class MovieSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CineSearch")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        
        # Custom color scheme
        self.bg_color = "#f0f0f0"  # Light gray background
        self.card_color = "#ffffff"  # Pure white cards
        self.accent_color = "#4a6fa5"  # Bright blue
        self.text_color = "#e63946"  # Vibrant red
        self.light_text = "#e63946"  # Dark gray for text
        self.bg_input_color="#fe7373"

        # Initialize movie data and search index
        self.all_movie_data = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.initialize_movie_data()

         
        # Input field colors
        self.input_bg = "#ffffff"
        self.input_fg = "#ffffff"
        self.input_highlight = "#fe7373"

                
        self.setup_ui()
        
    def initialize_movie_data(self):
        """Initialize movie data from TMDB API"""
        movie_ids = [
    603,    # The Matrix
    550,    # Fight Club
    27205,  # Inception
    634649, # Spider-Man: No Way Home
    102651, # Maleficent
    420809, # Maleficent: Mistress of Evil
    566574, # Phantom Thread
    566222, # The Post
    839327, # The Last Duel
    324857, # Spider-Man: Into the Spider-Verse
    961651, # The Marvels
    
    # New additions (20 more films)
    155,    # The Dark Knight
    19803,  # Eight Legged Freaks (2002) - Giant spiders attack town
    89492,  # Big Ass Spider! (2013) - Sci-fi comedy
    340837, # Sting (2024) - New Australian spider horror
    680,    # Pulp Fiction
    24428,  # The Avengers
    157336, # Interstellar
    238,    # The Godfather
    424,    # Schindler's List
    98,     # Gladiator
    497,    # The Green Mile
    122,    # The Lord of the Rings: The Return of the King
    13,     # Forrest Gump
    120,    # The Lord of the Rings: The Fellowship of the Ring
    1891,   # The Empire Strikes Back
    299534, # Avengers: Endgame
    278,    # The Shawshank Redemption
    429,    # The Good, the Bad and the Ugly
    372058, # Your Name (Japanese anime)
    496243, # Parasite (Korean film)
    19404,  # Dilwale Dulhania Le Jayenge (Bollywood)
    346698, # Barbie (2023)
    569094, # Spider-Man: Across the Spider-Verse
]
        self.all_movie_data = []
        
        for movie_id in movie_ids:
            movie_details = self.fetch_movie_details(movie_id)
            if movie_details:
                self.all_movie_data.extend(movie_details)
        
        # Initialize TF-IDF search index
        if self.all_movie_data:
            self.vectorizer, self.tfidf_matrix = self.create_tfidf_index(self.all_movie_data)
    
    def create_tfidf_index(self, movies):
        """Create a TF-IDF vectorizer and matrix from movie descriptions"""
        descriptions = [movie.get('overview', '') for movie in movies]
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        return vectorizer, tfidf_matrix
    
    def fetch_movie_details(self, movie_id):
        """Fetch movie details and cast from TMDB API"""
        movie_data = self.get_movie_data(movie_id)
        if movie_data is None:
            return None

        cast_data = self.get_cast(movie_id)
        if cast_data is None:
            return None
        
        description = movie_data.get("overview", "N/A")
        poster_path = movie_data.get("poster_path", "")
        actors = [actor['name'] for actor in cast_data.get('cast', [])]
        
        movie_details = {
            "title": movie_data.get("title", "N/A"),
            "overview": description,
            "actors": actors,
            "poster_path": poster_path,
            "id": movie_id
        }
        
        return [movie_details]
    
    def get_movie_data(self, movie_id):
        """Fetch movie data by movie_id from TMDB API"""
        url = f"{base_url}/movie/{movie_id}?api_key={api_key}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            logger.error(f"Failed to fetch movie data. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching movie data: {str(e)}")
        return None
    
    def get_cast(self, movie_id):
        """Fetch movie cast by movie_id from TMDB API"""
        url = f"{base_url}/movie/{movie_id}/credits?api_key={api_key}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            logger.error(f"Failed to fetch cast data. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching cast data: {str(e)}")
        return None
    
    def setup_ui(self):
        
        """Set up the user interface"""
        # Configure styles
        self.style = ThemedStyle(self.root)
        self.style.set_theme("equilux")
        # Configure scrollable area
        self.style.configure("TScrollbar", 
                        gripcount=0,
                        background=self.accent_color,
                        troughcolor=self.bg_color,
                        bordercolor=self.bg_color,
                        arrowcolor="white")

        # Card styling
        self.style.configure("Card.TFrame",
                        background=self.card_color,
                        relief="solid",
                        borderwidth=1,
                        bordercolor="#e0e0e0")
                
        # Custom style configurations
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.light_text, font=("Helvetica", 10))
        self.style.configure("Title.TLabel", font=("Helvetica", 20, "bold"), foreground=self.text_color)
        self.style.configure("Card.TFrame", background=self.card_color, relief="ridge", borderwidth=2)
        self.style.configure("Card.TLabel", background=self.card_color, foreground=self.light_text, font=("Helvetica", 10))
        self.style.configure("TButton", background=self.accent_color, foreground=self.light_text, 
                           font=("Helvetica", 10, "bold"), borderwidth=1)
        # Add this with your other style configurations
        self.style.configure("Similarity.Horizontal.TProgressbar",
                            thickness=6,
                            troughrelief='flat',
                            troughcolor=self.bg_color)
        self.style.map("TButton", 
                      background=[("active", self.text_color), ("pressed", self.text_color)])
        self.style.configure("TEntry", fieldbackground=self.bg_input_color , foreground=self.light_text, 
                           insertcolor=self.light_text, font=("Helvetica", 10))
        
        # Main container frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header section
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill="x", pady=(0, 20))
        
        self.title_label = ttk.Label(self.header_frame, text="CineSearch Pro", style="Title.TLabel")
        self.title_label.pack(side="top", pady=10)
        
        self.subtitle_label = ttk.Label(self.header_frame, 
                                      text="Discover movies by actor or description", 
                                      font=("Helvetica", 12))
        self.subtitle_label.pack(side="top")
        
        # Search section
        self.search_frame = ttk.Frame(self.main_frame)
        self.search_frame.pack(fill="x", pady=10)
        
        # Actor search
        self.actor_frame = ttk.Frame(self.search_frame)
        self.actor_frame.pack(side="left", expand=True, padx=10)
        
        self.actor_label = ttk.Label(self.actor_frame, text="Search by Actor:")
        self.actor_label.pack(anchor="w")
        
        input_style = ttk.Style()
        input_style.configure("Big.TEntry",
                            font=("Helvetica", 12),
                            fieldbackground="#fe7373",
                            foreground=self.input_fg,
                            padding=10,
                            bordercolor=self.accent_color,
                            lightcolor=self.input_highlight,
                            darkcolor=self.accent_color)
        
        self.actor_entry = ttk.Entry(self.actor_frame, width=30, style="Big.TEntry")
        self.actor_entry.pack(fill='x', pady=5, ipady=8)  # Increased internal padding
        
        
        # Description search
        self.desc_frame = ttk.Frame(self.search_frame)
        self.desc_frame.pack(side="left", expand=True, padx=10)
        
        self.desc_label = ttk.Label(self.desc_frame, text="Search by Description:")
        self.desc_label.pack(anchor="w")
        
        self.desc_entry = ttk.Entry(self.desc_frame, width=30, style="Big.TEntry")
        self.desc_entry.pack(fill='x', pady=5, ipady=8)
        self.desc_entry.bind("<Return>", lambda e: self.search_movies())
        
        
        # Search button
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill="x", pady=20)
        
        self.style.configure("Bright.TButton",
                       font=("Helvetica", 12, "bold"),
                       background=self.accent_color,
                       foreground="white",
                       borderwidth=2,
                       relief="raised",
                       padding=10)
    
        self.search_button = ttk.Button(self.button_frame, 
                                  text="Search Movies", 
                                  style="Bright.TButton",
                                  command=self.search_movies)
        self.search_button.pack(pady=15, ipadx=25, ipady=8)
        
        # Results section
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.pack(fill="both", expand=True)
        
        self.results_label = ttk.Label(self.results_frame, text="Search Results", 
                                     font=("Helvetica", 12, "bold"))
        self.results_label.pack(anchor="w", pady=(0, 10))
        
        # Canvas and scrollbar for results
        self.canvas = tk.Canvas(self.results_frame, bg=self.bg_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configure mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Initially empty results
        self.clear_results()
        
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def clear_results(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        empty_label = ttk.Label(self.scrollable_frame, 
                               text="Your search results will appear here", 
                               font=("Helvetica", 12))
        empty_label.pack(pady=50)
        
    def search_movies(self):

        actor_name = self.actor_entry.get().strip()
        description = self.desc_entry.get().strip()
        
        
        if not actor_name and not description:
            messagebox.showwarning("Input Error", "Please enter either an actor's name or a description.")
            return
            
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Show loading state
        loading_label = ttk.Label(self.scrollable_frame, text="Searching...", font=("Helvetica", 12))
        loading_label.pack(pady=50)
        self.root.update()  # Force UI update
        
        try:
            logger.debug(f"Starting search - Actor: '{actor_name}', Description: '{description}'")
            
            # Perform search
            if actor_name:
                results = self.search_by_actor(actor_name)
            # Set default similarity for actor searches
                for movie in results:
                    movie['similarity_score'] = 1.0  # Full match for actor searches
            else:
                results = self.search_by_description(description, top_n=5)
                
            # Remove loading label
            loading_label.destroy()
            
            # Display results
            if not results:
                logger.debug("No results found")
                no_results = ttk.Label(self.scrollable_frame, 
                                     text="No movies found matching your search", 
                                     font=("Helvetica", 12))
                no_results.pack(pady=50)
                return
            # Create container frame for centered results
            results_container = ttk.Frame(self.scrollable_frame)
            results_container.pack(expand=True, pady=20)
            
            # Configure grid to center items
            max_cols = 3
            for i in range(max_cols):
                results_container.grid_columnconfigure(i, weight=1)
            # Create a grid layout for results
            row, col = 0, 0
            
            for movie in results:
                logger.debug(f"Processing movie: {movie.get('title', 'Unknown')}")
                
                # Create card frame
                card = ttk.Frame(results_container, style="Card.TFrame", padding=15)  # Increased padding
                card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
                
                # Movie poster
                poster_path = movie.get('poster_path')
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
                    try:
                        logger.debug(f"Fetching poster from: {poster_url}")
                        response = requests.get(poster_url, timeout=5)
                        if response.status_code == 200:
                            image_data = Image.open(BytesIO(response.content))
                            image_data = image_data.resize((150, 225), Image.LANCZOS)
                            photo = ImageTk.PhotoImage(image_data)
                            
                            poster_label = tk.Label(card, image=photo, bg=self.card_color)
                            poster_label.image = photo
                            poster_label.pack(pady=(0, 10))
                        else:
                            logger.warning(f"Failed to fetch poster. Status code: {response.status_code}")
                            self.show_no_image(card)
                    except Exception as e:
                        logger.error(f"Error loading poster image: {str(e)}")
                        self.show_no_image(card)
                else:
                    logger.debug("No poster path available")
                    self.show_no_image(card)
                
                # Movie title
                title = movie.get('title', 'Unknown Title')
                logger.debug(f"Displaying movie: {title}")
                title_label = ttk.Label(card, text=title, style="Card.TLabel", 
                                      font=("Helvetica", 11, "bold"), wraplength=150)
                title_label.pack()
                
                # Movie description (truncated)
                desc = movie.get('overview', 'No description available')
                desc = (desc[:100] + '...') if len(desc) > 100 else desc
                desc_label = ttk.Label(card, text=desc, style="Card.TLabel", 
                                     wraplength=150, justify="left")
                desc_label.pack(pady=(5, 0))
                
                # Update grid position
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
                
                if 'similarity_score' in movie:
                    self.add_similarity_display(card, movie['similarity_score'])
            # Configure grid weights for responsive layout
            for i in range(max_cols):
                self.scrollable_frame.grid_columnconfigure(i, weight=1)
                
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            loading_label.destroy()
            messagebox.showerror("Search Error", f"An error occurred during search:\n{str(e)}")
            self.clear_results()
    
    def add_similarity_display(self, parent_frame, score):
        """Adds a visual similarity indicator to movie cards"""
        score_percent = min(max(score * 100, 0), 100)  # Ensure 0-100 range
        
        # Container frame
        sim_frame = ttk.Frame(parent_frame)
        sim_frame.pack(fill='x', pady=(5, 0))
        
        # Label
        ttk.Label(sim_frame, 
                text=f"Match: {score_percent:.1f}%", 
                style="Card.TLabel",
                font=("Helvetica", 8)).pack(side='left')
        
        # Progress bar with color coding
        style_name = f"Similarity.Horizontal.TProgressbar"
        self.style.configure(style_name, 
                            troughcolor=self.card_color,
                            background=self.text_color if score > 0.7 
                                    else self.accent_color if score > 0.3 
                                    else "#666666")
        
        progress = ttk.Progressbar(sim_frame, 
                                orient='horizontal',
                                length=80, 
                                mode='determinate',
                                style=style_name)
        progress.pack(side='left', padx=5)
        progress['value'] = score_percent
    def show_no_image(self, parent):
        no_image = ttk.Label(parent, text="No Image Available", 
                            style="Card.TLabel", width=20, height=10)
        no_image.pack(pady=(0, 10))

    def search_by_actor(self, actor_name):
        """Search for movies by actor's name with case-insensitive matching."""
        logger.debug(f"Starting actor search for: {actor_name}")
        results = []
        
        if not self.all_movie_data:
            logger.warning("No movies data available for search")
            return results
            
        actor_name_lower = actor_name.lower().strip()
        
        for movie in self.all_movie_data:
            try:
                # Get actors list or empty list if not available
                actors = movie.get('actors', [])
                
                # Check if actor is in the movie's cast (case-insensitive)
                if any(actor_name_lower in actor.lower() for actor in actors):
                    results.append(movie)
            except Exception as e:
                logger.error(f"Error processing movie {movie.get('title', 'Unknown')}: {str(e)}")
                continue
                
        logger.debug(f"Found {len(results)} movies for actor {actor_name}")
        return results

    def search_by_description(self, query, top_n=5):
        """Search movies by description using TF-IDF and cosine similarity"""
        if not query or not self.all_movie_data:
            return []

        try:
            # Create fresh TF-IDF vectorizer to avoid any array comparison issues
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Get all movie descriptions
            descriptions = [movie.get('overview', '') for movie in self.all_movie_data]
            
            # Fit and transform the movie descriptions
            tfidf_matrix = vectorizer.fit_transform(descriptions)
            
            # Transform the query
            query_vec = vectorizer.transform([query])
            
            # Calculate cosine similarities (returns a 2D array)
            similarities = cosine_similarity(tfidf_matrix, query_vec)
            
            # Convert to 1D array and get indices sorted by similarity
            similarity_scores = similarities.flatten()
            sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
            
            if len(similarity_scores) > 0:
                max_score = similarity_scores.max()
                if max_score > 0:
                    similarity_scores = similarity_scores / max_score
            # Collect results above threshold
            threshold = 0.1
            results = []
            for idx in sorted_indices:
                score = similarity_scores[idx]
                if score > threshold and len(results) < top_n:
                    movie = self.all_movie_data[idx].copy()
                    movie['similarity_score'] = float(score)  # Convert to Python float
                    results.append(movie)
            
            return results

        except Exception as e:
            logger.error(f"Error in description search: {str(e)}")
            messagebox.showerror("Search Error", f"An error occurred during search:\n{str(e)}")
            return []
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieSearchApp(root)
    root.mainloop()