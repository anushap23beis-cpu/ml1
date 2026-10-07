# Movie Recommendation System — Report Matched Version

This version is structured to match the submitted Machine Learning-II report.

## Required files
Place these files in `data/`:
- movies.csv
- ratings.csv

Expected columns:
- movies.csv: movieId, title, genres
- ratings.csv: userId, movieId, rating, timestamp

## Run
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Main report-aligned components
- MovieLens dataset
- preprocessing
- TF-IDF
- cosine similarity
- content-based filtering
- collaborative filtering
- hybrid recommendation
- data analysis and visualization
- reported evaluation metrics
- Streamlit deployment
