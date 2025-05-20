import streamlit as st
import numpy as np
import faiss
from langchain.document_loaders import CSVLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

class IMDBot:
    def __init__(self, data_path="imdb.csv"):
        """Initialize with data processing and recommendation systems"""
        self.documents = self._load_data(data_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index, self.embeddings = self._build_index()

    def _load_data(self, path):
        """Create enriched movie documents with comprehensive metadata"""
        loader = CSVLoader(file_path=path)
        raw_docs = loader.load()

        processed_docs = []
        for doc in raw_docs:
            content = doc.page_content
            lines = content.split('\n')
            metadata = {line.split(':')[0].strip(): line.split(':')[1].strip()
                       for line in lines if ':' in line}

            # Sentiment analysis
            description = metadata.get('Description', '')
            sentiment = TextBlob(description).sentiment
            sentiment_label = 'positive' if sentiment.polarity > 0.2 else \
                             'negative' if sentiment.polarity < -0.2 else 'neutral'

            # Store only necessary content for search
            enhanced_content = (
                f"Title: {metadata.get('Title', '')}\n"
                f"Genre: {metadata.get('Genre', '')}\n"
                f"Plot: {description}\n"
                f"Director: {metadata.get('Director', '')}\n"
                f"Actors: {metadata.get('Actors', '')}\n"
                f"Year: {metadata.get('Year', '')}\n"
                f"Rating: {metadata.get('Rating', '')}\n"
                f"Votes: {metadata.get('Votes', '')}\n"
                f"Revenue: {metadata.get('Revenue (Millions)', '')} million\n"
                f"Metascore: {metadata.get('Metascore', '')}\n"
                f"Runtime: {metadata.get('Runtime (Minutes)', '')} minutes\n"
                f"Sentiment: {sentiment_label} ({sentiment.polarity:.2f})\n"
            )
            processed_docs.append(Document(
                page_content=enhanced_content,
                metadata={
                    **metadata,
                    'sentiment': sentiment.polarity,
                    'sentiment_label': sentiment_label,
                    'subjectivity': sentiment.subjectivity
                }
            ))
        return processed_docs

    def _build_index(self):
        """Build FAISS index for content embeddings"""
        content_embeddings = self.model.encode([doc.page_content for doc in self.documents])
        content_embeddings = np.array(content_embeddings).astype("float32")
        content_index = faiss.IndexFlatL2(content_embeddings.shape[1])
        content_index.add(content_embeddings)
        return content_index, content_embeddings

    def _get_recommendation(self, query, genre_filter=None, min_rating=None, top_n=3):
        """Recommendation combining semantic search, genre, and quality metrics"""
        query_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, len(self.documents))

        candidates = []
        for idx, distance in zip(indices[0], distances[0]):
            doc = self.documents[idx]
            metadata = doc.metadata

            # Extract key metrics
            rating = float(metadata.get('Rating', 0))
            genres = metadata.get('Genre', '').lower()
            sentiment = float(metadata.get('sentiment', 0))

            # Apply filters
            if genre_filter and genre_filter.lower() not in genres:
                continue
            if min_rating and rating < min_rating:
                continue

            # Calculate composite score
            semantic_score = 1 / (1 + distance)
            composite_score = (
                0.5 * rating / 10 +
                0.3 * (sentiment + 1) / 2 +
                0.2 * semantic_score
            )
            candidates.append((composite_score, doc))

        # Return top results
        candidates.sort(reverse=True, key=lambda x: x[0])
        return [doc for (score, doc) in candidates[:top_n]]

    def _generate_recommendation_response(self, movies):
        """Create concise recommendation responses"""
        responses = []
        for movie in movies:
            metadata = movie.metadata
            title = metadata.get('Title', 'Unknown Movie')
            year = metadata.get('Year', '')
            rating = metadata.get('Rating', '')
            genre = metadata.get('Genre', '')
            sentiment = metadata.get('sentiment_label', '').capitalize() + f" ({metadata.get('sentiment', 0):.2f})"
            plot = metadata.get('Description', '')[:120] + "..."
            responses.append(
                f"🎬 {title} ({year})\n"
                f"⭐ Rating: {rating}/10 | 🎭 {genre} | 😊 {sentiment}\n"
                f"📖 {plot}\n"
                f"── ⋆⋅☆⋅⋆ ──"
            )
        return "\n\n".join(responses)

    def search(self, query):
        """Search with specific attribute retrieval and recommendations"""
        try:
            query_lower = query.lower().strip()

            # Handle recommendation queries
            if any(word in query_lower for word in ['best', 'top', 'recommend', 'suggest']):
                genre_keywords = ['thriller', 'comedy', 'horror', 'sci-fi', 'drama', 'action', 'romance', 'adventure', 'crime', 'music']
                genre_filter = next((word for word in genre_keywords if word in query_lower), None)
                min_rating = None
                if 'rating greater than' in query_lower:
                    try:
                        min_rating = float(query_lower.split('rating greater than')[-1].strip().split()[0])
                    except:
                        min_rating = 8.0 if 'best' in query_lower else 7.0
                else:
                    min_rating = 8.0 if 'best' in query_lower else 7.0

                results = self._get_recommendation(query, genre_filter, min_rating, top_n=3)
                if results:
                    header = f"🔍 Based on your request for "
                    header += f"{genre_filter} movies" if genre_filter else "top movie recommendations"
                    header += f" (minimum rating: {min_rating}/10):\n\n"
                    return f"{header}{self._generate_recommendation_response(results)}"
                else:
                    return "Couldn't find movies matching all criteria. Try broadening your search."

            # Handle specific attribute queries
            keywords = {
                "director": "Director",
                "cast": "Actors",
                "actor": "Actors",
                "actors": "Actors",
                "actress": "Actors",
                "who directed": "Director",
                "who is director": "Director",
                "who is the director": "Director",
                "who is in": "Actors",
                "who stars": "Actors",
                "rating": "Rating",
                "rate": "Rating",
                "imdb rating": "Rating",
                "votes": "Votes",
                "revenue": "Revenue (Millions)",
                "metascore": "Metascore",
                "runtime": "Runtime (Minutes)",
                "genre": "Genre",
                "description": "Description",
                "plot": "Description",
                "year": "Year"
            }

            for keyword, field in keywords.items():
                if keyword in query_lower:
                    movie_name = query_lower.replace(keyword, "").strip(" ?")
                    if not movie_name:
                        return "Please specify a movie title."
                    query_embedding = self.model.encode([movie_name]).astype("float32")
                    distances, indices = self.index.search(query_embedding, 1)
                    doc = self.documents[indices[0][0]]
                    answer = doc.metadata.get(field, "Not found")
                    title = doc.metadata.get("Title", "Unknown Movie")
                    unit = " million" if field == "Revenue (Millions)" else " minutes" if field == "Runtime (Minutes)" else ""
                    return f"{field} of '{title}' is: {answer}{unit}"

            return "Sorry, I couldn't understand or answer this question."

        except Exception as e:
            return f"Error processing request - {str(e)}"

# Streamlit app interface
st.title("IMDBot: Movie Information and Recommendations")
st.write("Ask about a movie's rating, director, cast, genre, runtime, description, year, votes, revenue, or metascore, or request movie recommendations!")

# File uploader for the CSV
uploaded_file = st.file_uploader("Upload your IMDB CSV file", type=["csv"])
if uploaded_file is not None:
    # Save the uploaded file to disk
    with open("imdb.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize the bot
    bot = IMDBot("imdb.csv")

    # Input query
    query = st.text_input("Enter your question (e.g., 'What is the rating of Cars?' or 'Suggest a romance movie')")
    
    if query:
        # Get and display the response
        response = bot.search(query)
        st.write(response)
else:
    st.write("Please upload the IMDB CSV file to start.")