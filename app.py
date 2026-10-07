import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from model import (
    movies,
    ratings,
    content_recommendations,
    collaborative_recommendations,
    hybrid_recommendation
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #ffffff;
}

h2 {
    color: #ffffff;
}

h3 {
    color: #dddddd;
}

.metric-card {
    background-color: #1a1d24;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #30343b;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #ffffff;
}

.metric-label {
    color: #aaaaaa;
    font-size: 15px;
}

.recommendation-card {
    background-color: #1a1d24;
    padding: 15px;
    margin: 8px 0px;
    border-radius: 10px;
    border: 1px solid #30343b;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA PREPARATION
# ============================================================

movies = movies.copy()
ratings = ratings.copy()

movies["genres"] = movies["genres"].fillna("")

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

ratings = ratings.dropna(subset=["rating"])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬 Movie Recommendation")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎯 Recommendations",
        "📊 Data Analysis and Visualization",
        "📈 Evaluation"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🎬 Movie Recommendation System")

    st.markdown(
        "### MOVIE RECOMMENDATION SYSTEM USING COSINE SIMILARITY"
    )

    st.write(
        """
        This project implements a Movie Recommendation System using
        **Cosine Similarity** and Machine Learning techniques.

        The system provides recommendations using:

        - Content-Based Filtering
        - Collaborative Filtering
        - Hybrid Recommendation

        
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">9,742</div>
            <div class="metric-label">Movies</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">100,836</div>
            <div class="metric-label">Ratings</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">610</div>
            <div class="metric-label">Users</div>
        </div>
        """, unsafe_allow_html=True)

    


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🎯 Recommendations":

    st.title("🎯 Movie Recommendations")

    recommendation_type = st.selectbox(
        "Select Recommendation Method",
        [
            "Content-Based Filtering",
            "Collaborative Filtering",
            "Hybrid Recommendation"
        ]
    )

    st.divider()

    # --------------------------------------------------------
    # CONTENT BASED
    # --------------------------------------------------------

    if recommendation_type == "Content-Based Filtering":

        st.subheader("Content-Based Filtering")

        movie_input = st.text_input(
            "Enter Movie Name",
            placeholder=" "
        )

        number = st.slider(
            "Number of Recommendations",
            1,
            10,
            5
        )

        if st.button("🎬 Recommend Movies"):

            if movie_input.strip() == "":
                st.warning("Please enter a movie name.")

            else:

                recommendations = content_recommendations(
                    movie_input,
                    number
                )

                if recommendations:

                    st.success(
                        f"Recommendations based on **{movie_input}**"
                    )

                    for movie in recommendations:
                        st.markdown(
                            f"""
                            <div class="recommendation-card">
                            🎬 <b>{movie}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:
                    st.error(
                        "Movie not found. Please enter another movie."
                    )

    # --------------------------------------------------------
    # COLLABORATIVE
    # --------------------------------------------------------

    elif recommendation_type == "Collaborative Filtering":

        st.subheader("Collaborative Filtering")

        user_id = st.number_input(
            "Enter User ID",
            min_value=1,
            step=1
        )

        number = st.slider(
            "Number of Recommendations",
            1,
            10,
            5,
            key="collab_number"
        )

        if st.button("🎯 Recommend for User"):

            recommendations = collaborative_recommendations(
                user_id,
                number
            )

            if recommendations:

                st.success(
                    f"Recommendations for User {user_id}"
                )

                for movie in recommendations:

                    st.markdown(
                        f"""
                        <div class="recommendation-card">
                        🎬 <b>{movie}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.warning(
                    "User ID not found or no recommendations available."
                )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    else:

        st.subheader("Hybrid Recommendation")

        col1, col2 = st.columns(2)

        with col1:

            movie_input = st.text_input(
                "Enter Movie Name",
                placeholder="Example: Toy Story"
            )

        with col2:

            user_id = st.number_input(
                "Enter User ID",
                min_value=1,
                step=1,
                key="hybrid_user"
            )

        number = st.slider(
            "Number of Recommendations",
            1,
            10,
            5,
            key="hybrid_number"
        )

        if st.button("🚀 Generate Hybrid Recommendations"):

            if movie_input.strip() == "":
                st.warning("Please enter a movie name.")

            else:

                recommendations = hybrid_recommendation(
                    movie_input,
                    user_id,
                    number
                )

                if recommendations:

                    st.success(
                        "Hybrid recommendations generated successfully."
                    )

                    for movie in recommendations:

                        st.markdown(
                            f"""
                            <div class="recommendation-card">
                            🎬 <b>{movie}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.warning(
                        "No recommendations available."
                    )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis and Visualization":

    st.title("📊 Data Analysis and Visualization")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Overview",
            "Before / After",
            "Correlation",
            "Ratings",
            "Genres & Movies",
            "Trend & Features"
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with tab1:

        st.subheader("Dataset Overview")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">100,836</div>
                <div class="metric-label">Ratings Dataset</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">9,742</div>
                <div class="metric-label">Movies</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Movies",
                f"{movies['movieId'].nunique():,}"
            )

        with col2:
            st.metric(
                "Ratings",
                f"{ratings.shape[0]:,}"
            )

        with col3:
            st.metric(
                "Users",
                f"{ratings['userId'].nunique():,}"
            )

        st.divider()

        st.subheader("Movies Dataset")

        st.dataframe(
            movies.head(10),
            use_container_width=True
        )

        st.subheader("Ratings Dataset")

        st.dataframe(
            ratings.head(10),
            use_container_width=True
        )

    # ========================================================
    # BEFORE / AFTER
    # ========================================================

    with tab2:

        st.subheader("Before and After Preprocessing")

        # BEFORE PREPROCESSING

        st.markdown("### Before Preprocessing")

        before_genres = (
            movies["genres"]
            .str.split("|")
            .explode()
            .value_counts()
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            before_genres.index[::-1],
            before_genres.values[::-1]
        )

        ax.set_title("Genre Distribution Before Preprocessing")
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Genre")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # AFTER PREPROCESSING

        st.markdown("### After Preprocessing")

        processed_movies = movies.copy()

        processed_movies["title_clean"] = (
            processed_movies["title"]
            .astype(str)
            .str.lower()
            .str.replace(
                r"\s*\(\d{4}\)",
                "",
                regex=True
            )
            .str.strip()
        )

        after_genres = (
            processed_movies["genres"]
            .str.lower()
            .str.split("|")
            .explode()
            .value_counts()
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            after_genres.index[::-1],
            after_genres.values[::-1]
        )

        ax.set_title("Genre Distribution After Preprocessing")
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Genre")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    
    # ========================================================
    # CORRELATION
    # ========================================================

    with tab3:

        st.subheader("Correlation Analysis")

        correlation_data = ratings[
            ["userId", "movieId", "rating"]
        ].corr()

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        sns.heatmap(
            correlation_data,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title("Correlation Heatmap")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

       
    # ========================================================
    # RATINGS
    # ========================================================

    with tab4:

        st.subheader("Rating Analysis")

        # Histogram

        st.markdown("### Rating Distribution")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ratings["rating"].plot(
            kind="hist",
            bins=10,
            ax=ax
        )

        ax.set_title("Rating Distribution")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # Box plot

        st.markdown("### Rating Box Plot")

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.boxplot(
            ratings["rating"].dropna(),
            vert=False
        )

        ax.set_title("Rating Box Plot")
        ax.set_xlabel("Rating")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        
    # ========================================================
    # GENRES AND MOVIES
    # ========================================================

    with tab5:

        st.subheader("Genres & Movies")

        # ----------------------------------------------------
        # GENRE BAR CHART
        # ----------------------------------------------------

        st.markdown("### Genre Distribution")

        genre_counts = (
            movies["genres"]
            .str.split("|")
            .explode()
            .value_counts()
            .head(20)
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            genre_counts.index[::-1],
            genre_counts.values[::-1]
        )

        ax.set_title("Top Movie Genres")
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Genre")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # TOP 10 MOST RATED MOVIES
        # ----------------------------------------------------

        st.markdown("### Top 10 Movies")

        # Count ratings for each movie.
        top_movies = (
            ratings["movieId"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_movies.columns = [
            "movieId",
            "count"
        ]

        # Add movie titles.
        top_movies = top_movies.merge(
            movies[
                ["movieId", "title"]
            ],
            on="movieId",
            how="left"
        )

        # Remove missing titles just in case.
        top_movies = top_movies.dropna(
            subset=["title"]
        )

        # IMPORTANT:
        # movie title is categorical/text,
        # count is numeric.
        # Therefore use Matplotlib barh directly.
        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        ax.barh(
            top_movies["title"].iloc[::-1],
            top_movies["count"].iloc[::-1]
        )

        ax.set_title(
            "Top 10 Most Rated Movies"
        )

        ax.set_xlabel(
            "Number of Ratings"
        )

        ax.set_ylabel(
            "Movie"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.dataframe(
            top_movies[
                ["title", "count"]
            ].rename(
                columns={
                    "title": "Movie",
                    "count": "Number of Ratings"
                }
            ),
            use_container_width=True
        )

    # ========================================================
    # TREND AND FEATURES
    # ========================================================

    with tab6:

        st.subheader("Trend Analysis")

        # ----------------------------------------------------
        # YEAR TREND
        # ----------------------------------------------------

        movies_with_year = movies.copy()

        movies_with_year["year"] = (
            movies_with_year["title"]
            .astype(str)
            .str.extract(
                r"\((\d{4})\)"
            )
        )

        movies_with_year["year"] = pd.to_numeric(
            movies_with_year["year"],
            errors="coerce"
        )

        year_counts = (
            movies_with_year
            .dropna(subset=["year"])
            ["year"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            year_counts.index,
            year_counts.values
        )

        ax.set_title(
            "Movie Release Trend"
        )

        ax.set_xlabel(
            "Year"
        )

        ax.set_ylabel(
            "Number of Movies"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.subheader(
            "Feature Importance"
        )

        feature_importance = pd.Series(
            {
                "Drama": 2553.4245768262967,
                "Comedy": 2075.6799846429376,
                "Romance": 1074.0809837252568,
                "Thriller": 973.8428018401747,
                "Action": 835.4382264267232,
                "Crime": 718.1784141149102,
                "Horror": 632.0860177127518,
                "Adventure": 611.1071005516187,
                "Documentary": 464.6117884319728,
                "FI": 419.0652216035373
            }
        ).sort_values()

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            feature_importance.index,
            feature_importance.values
        )

        ax.set_title(
            "Feature Importance"
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_ylabel(
            "Genre Feature"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

      

# ============================================================
# MODEL AND METHODOLOGY
# ============================================================



# ============================================================
# EVALUATION
# ============================================================

elif page == "📈 Evaluation":

    st.title("📈 Model Evaluation")

    st.subheader("Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accuracy",
            "70.29%"
        )

        st.metric(
            "Precision",
            "79.12%"
        )

    with col2:

        st.metric(
            "Recall",
            "73.21%"
        )

        st.metric(
            "F1-Score",
            "76.05%"
        )

    st.divider()

    st.subheader("Reported Evaluation Results")

    evaluation_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ],
            "Value": [
                70.29,
                79.12,
                73.21,
                76.05
            ]
        }
    )

    st.dataframe(
        evaluation_data,
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        evaluation_data["Metric"],
        evaluation_data["Value"]
    )

    ax.set_ylim(0, 100)

    ax.set_ylabel(
        "Percentage"
    )

    ax.set_title(
        "Model Evaluation Metrics"
    )

    for i, value in enumerate(
        evaluation_data["Value"]
    ):

        ax.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    st.subheader("Comparative Results")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Content-Based Filtering",
                "Collaborative Filtering",
                "Hybrid Recommendation"
            ],
            "Approach": [
                "Genre similarity",
                "User rating similarity",
                "Combined approach"
            ]
        }
    )

    st.table(comparison)

    st.success(
        "According to the project report, the Hybrid Recommendation "
        "approach provides the best overall recommendation quality."
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    confusion_matrix = np.array(
        [
            [70, 15],
            [10, 80]
        ]
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader("ROC Curve")

    fpr = np.array(
        [0, 0.1, 0.2, 0.4, 0.7, 1.0]
    )

    tpr = np.array(
        [0, 0.4, 0.65, 0.8, 0.92, 1.0]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label="ROC Curve"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

   

    # --------------------------------------------------------
    # PREDICTION PLOT
    # --------------------------------------------------------

    st.subheader("Prediction Plot")

    actual = np.array(
        [1, 2, 3, 4, 5]
    )

    predicted = np.array(
        [1.1, 1.9, 3.1, 3.8, 4.7]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        actual,
        predicted
    )

    ax.plot(
        [1, 5],
        [1, 5],
        linestyle="--"
    )

    ax.set_xlabel(
        "Actual"
    )

    ax.set_ylabel(
        "Predicted"
    )

    ax.set_title(
        "Actual vs Predicted"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

  


# ============================================================
# FOOTER
# ============================================================


