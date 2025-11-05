import streamlit as st
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Helper functions
# --------------------------
def prepare_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['Genre'] + ' ' + x['Tags'] + ' ' + x['Actors'] + ' ' + x['ViewerRating']

def get_recommendations(title, cosine_sim):
    title = title.replace(' ', '').lower()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    result = netflix_data.iloc[movie_indices]
    result.reset_index(inplace=True)
    return result

# --------------------------
# Load and prepare dataset
# --------------------------
st.title("🎬 Netflix Movie Recommender")

netflix_data = pd.read_csv('NetflixDataset.csv', encoding='latin-1', index_col='Title')
netflix_data.index = netflix_data.index.str.title()
netflix_data = netflix_data[~netflix_data.index.duplicated()]
netflix_data.rename(columns={'View Rating': 'ViewerRating'}, inplace=True)

Language = netflix_data.Languages.str.get_dummies(',')
Lang = set(Language.columns.str.strip().values.tolist())
Titles = set(netflix_data.index.to_list())

netflix_data['Genre'] = netflix_data['Genre'].astype('str')
netflix_data['Tags'] = netflix_data['Tags'].astype('str')
netflix_data['IMDb Score'] = netflix_data['IMDb Score'].apply(lambda x: 6.6 if math.isnan(x) else x)
netflix_data['Actors'] = netflix_data['Actors'].astype('str')
netflix_data['ViewerRating'] = netflix_data['ViewerRating'].astype('str')

new_features = ['Genre', 'Tags', 'Actors', 'ViewerRating']
selected_data = netflix_data[new_features]

for new_feature in new_features:
    selected_data.loc[:, new_feature] = selected_data.loc[:, new_feature].apply(prepare_data)

selected_data.index = selected_data.index.str.lower().str.replace(" ", "")
selected_data['soup'] = selected_data.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(selected_data['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
selected_data.reset_index(inplace=True)
indices = pd.Series(selected_data.index, index=selected_data['Title'])

# --------------------------
# Streamlit UI
# --------------------------
st.sidebar.header("Select Options")

selected_titles = st.sidebar.multiselect("🎥 Choose a movie", sorted(list(Titles)))
selected_languages = st.sidebar.multiselect("🌐 Choose languages", sorted(list(Lang)))

if st.sidebar.button("🔍 Recommend"):
    if selected_titles and selected_languages:
        df_final = pd.DataFrame()
        for moviename in selected_titles:
            result = get_recommendations(moviename, cosine_sim2)
            for language in selected_languages:
                filtered = result[result['Languages'].str.count(language) > 0]
                df_final = pd.concat([df_final, filtered], ignore_index=True)

        df_final.drop_duplicates(subset='Title', inplace=True)
        df_final.sort_values(by='IMDb Score', ascending=False, inplace=True)

        st.subheader("🎯 Recommended Movies:")
        for i, row in df_final.iterrows():
            st.image(row['Image'], width=150)
            st.write(f"**{row['Title']}**")
            st.caption(f"🎭 {row['Genre']} | ⭐ {row['IMDb Score']}")
            st.divider()
    else:
        st.warning("Please select at least one movie and language.")
