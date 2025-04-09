import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Login Manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# TMDB API
TMDB_API_KEY = 'fd63624ffe4ce8868bd42a144141ba75'

# Load datasets
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

# Load user
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# TMDB helper

def get_movie_details(title):
    url = f'https://api.themoviedb.org/3/search/movie'
    params = {'api_key': TMDB_API_KEY, 'query': title}
    response = requests.get(url, params=params).json()

    if response['results']:
        movie = response['results'][0]
        return {
            'poster': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else "",
            'overview': movie.get('overview', ""),
            'vote_average': movie.get('vote_average', "N/A")
        }
    return {'poster': '', 'overview': '', 'vote_average': 'N/A'}

# Recommendation logic
def recommend_movies_for_user(user_id, top_n=5):
    user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    cosine_sim = cosine_similarity(user_movie_matrix)
    user_index = user_id - 1

    similar_users = sorted(
        list(enumerate(cosine_sim[user_index])),
        key=lambda x: x[1], reverse=True
    )[1:top_n+1]

    similar_user_ids = [user[0] + 1 for user in similar_users]
    recommended_movies = ratings_df[ratings_df['userId'].isin(similar_user_ids)]
    movie_recommendations = recommended_movies.groupby('movieId')['rating'].mean().reset_index()
    movie_recommendations = movie_recommendations.sort_values(by='rating', ascending=False).head(top_n)
    movie_recommendations = pd.merge(movie_recommendations, movies_df, on='movieId')

    result = []
    for _, row in movie_recommendations.iterrows():
        tmdb = get_movie_details(row['title'])
        result.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'rating': round(row['rating'], 2),
            'poster': tmdb['poster'],
            'overview': tmdb['overview'],
            'vote_average': tmdb['vote_average']
        })
    return result

# Routes
@app.route('/')
def home():
    genres = movies_df['genres'].str.split('|').explode().unique()
    return render_template('index.html', genres=genres)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/recommend/user/<int:user_id>')
@login_required
def recommend_for_user(user_id):
    recommendations = recommend_movies_for_user(user_id, top_n=5)
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/search')
def search():
    title = request.args.get('title')
    movie_details = get_movie_details(title)
    return jsonify(movie_details)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)