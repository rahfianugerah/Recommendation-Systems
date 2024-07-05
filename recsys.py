# %% [markdown]
# # Machine Learning Project - Book Recomendation Systems
# 
# - Name: Naufal Rahfi Anugerah
# - Dicoding ID: nrahfi
# - E-mail: 189nrahfi@gmail.com

# %% [markdown]
# ## Import Library

# %% [markdown]
# Importing all required library for this project

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import zipfile

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Data Gathering

# %% [markdown]
# ### Download Dataset

# %% [markdown]
# Download dataset using kaggle

# %%
!kaggle datasets download -d ruchi798/bookcrossing-dataset

# %% [markdown]
# ### Extract Downloaded File

# %%
# Extract zipped file using zippfile built-in function
filezip = "bookcrossing-dataset.zip" # variable for zipped file path
zip = zipfile.ZipFile(filezip, 'r') # read zipped file
zip.extractall() # extracting file
zip.close() # close zip file

# %% [markdown]
# ### Data Loading

# %% [markdown]
# Load the dataset

# %%
df = pd.read_csv("Books Data with Category Language and Summary\Preprocessed_data.csv")
df.head(len(df))

# %% [markdown]
# ## Data Assesing

# %% [markdown]
# Checking information about the dataset

# %%
df.info()

# %% [markdown]
# Assesing the dataset

# %%
# Function for assesing data
def data_assesing(data):
    # Display the total number of NaN and Null values in each column, sorted in descending order
    print(f"Total NaN/Null Data per Column:\n{data.isna().sum().sort_values(ascending=False)}\n")
    # Display the shape of the dataset
    print(f"Data Shape:\n{data.shape}")
    # Total duplicted data in dataset
    print(f"\nTotal Duplicated Data: {data.duplicated().sum()}")
# Call the function for assesing dataset hour.csv
data_assesing(df)

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# Discard records that have a not available value

# %%
df = pd.DataFrame(df.dropna())
df.head(len(df))

# %% [markdown]
# The dataset has been cleaned

# %%
data_assesing(df)

# %% [markdown]
# Discard another unused columns

# %%
df = pd.DataFrame(df.drop(columns=['img_s', 'img_l', 'img_m', 'Unnamed: 0', 'user_id']))

# %%
df.head()

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# This code below will create the visualization for rating distribution

# %%
# Calculate the count of each rating value
rating_counts = df['rating'].value_counts()

# Get rating values as x and their counts as y
ratings = rating_counts.index
count = rating_counts.values

# Create a plot using matplotlib
plt.figure(figsize=(12, 6))
plt.bar(ratings, count, color='skyblue')

# Add title and axis labels
plt.title('Rating Distribution', size=10)
plt.xlabel('Rating', size=10)
plt.ylabel('Count', size=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid lines with transparency (alpha=0.7)

# Display the plot
plt.show()


# %% [markdown]
# From the results of the visualization above, it can be concluded that the most books are books that have a rating of 0. Most likely this book has not been read by many people and is not popular among readers.

# %% [markdown]
# This code below will create the visualization for cleaned rating distribution

# %%
# Filter out rows where rating is 0
data_rating = df[df['rating'] != 0]

# Count values of 'rating'
rating_counts = data_rating['rating'].value_counts().sort_index()

# Plot using matplotlib
plt.figure(figsize=(12, 6))

# Create bars
plt.bar(rating_counts.index, rating_counts.values, color='skyblue')

plt.title('Rating Distribution (Cleaned)', size=10)
plt.xlabel('Rating', size=10)
plt.ylabel('Count', size=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid lines with transparency (alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# By removing the 0 rating, The most read book is a book with a rating of 8. It is likely that this book is very popular but does not have as high a rating as a book with a rating of 10.

# %% [markdown]
# Below is the code that can visualize the top 10 most readed books

# %%
# Count occurrences of each book and select top 10 most readed books
data_authors = df['book_title'].value_counts().head(10).reset_index()
data_authors.columns = ['book_title', 'count']

# Plotting using seaborn and matplotlib
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='book_title', data=data_authors, color='skyblue')

# Customizing labels and title
plt.xlabel('Count', size=10)
plt.ylabel('Author', size=10)
plt.title('Top 10 Most Readed Books', size=10)

# Adjusting tick label size for better readability
plt.xticks(size=10)
plt.yticks(size=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid lines with transparency (alpha=0.7)

# Displaying the plot
plt.show()

# %% [markdown]
# From the visualization above, can be concluded that book with the tile of 'Wild Animus' is the most popular book with more than 2000 readers

# %% [markdown]
# Below is the code to show top 10 years of publication

# %%
# Count occurrences of each year and select top 10
data_year = df['year_of_publication'].astype(int).astype(str).value_counts().head(10).reset_index()
data_year.columns = ['year', 'count']
data_year['year'] = 'Year ' + data_year['year']  # Adding 'Year ' prefix for better labeling

# Plotting using seaborn and matplotlib
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='year', data=data_year, color='skyblue')

# Customizing labels and title
plt.xlabel('Count', size=10)
plt.ylabel('Year of Publication', size=10)
plt.title('Top 10 Years of Publication', size=10)

# Adjusting tick label size for better readability
plt.xticks(size=10)
plt.yticks(size=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid lines with transparency (alpha=0.7)

# Displaying the plot
plt.show()

# %% [markdown]
# Can concluded, The year when the most books were released was 2002, with more than 80000 books.

# %% [markdown]
# This code below with visualize top 10 best-selling author

# %%
# Count occurrences of each book author and select top 10
data_authors = df['book_author'].value_counts().head(10).reset_index()
data_authors.columns = ['book_author', 'count']

# Plotting using seaborn and matplotlib
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='book_author', data=data_authors, color='skyblue')

# Customizing labels and title
plt.xlabel('Count', size=10)
plt.ylabel('Author', size=10)
plt.title('Top 10 Book Authors', size=10)

# Adjusting tick label size for better readability
plt.xticks(size=10)
plt.yticks(size=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid lines with transparency (alpha=0.7)

# Displaying the plot
plt.show()

# %% [markdown]
# It can be concluded that stephen king is the best-selling author in a certain period. Most likely the book he wrote has the most interesting story among other authors.

# %% [markdown]
# Below is the code to visualize the top 10 book publishers.

# %%
# Count occurrences of each book publisher and select top 10
data_publisher = df['publisher'].value_counts().head(10).reset_index()
data_publisher.columns = ['publisher', 'count']

# Plotting using seaborn and matplotlib
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='publisher', data=data_publisher, color='skyblue')

# Customizing labels and title
plt.xlabel('Count', size=10)
plt.ylabel('Author', size=10)
plt.title('Top 10 Book Publisher', size=10)

# Adjusting tick label size for better readability
plt.xticks(size=10)
plt.yticks(size=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid lines with transparency (alpha=0.7)

# Displaying the plot
plt.show()

# %% [markdown]
# It can be concluded that Ballentine Books was the largest publisher of books in its time, meaning that this publisher had a major contribution to the book industry in its time.

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# The following is a dataset that has gone through the data cleaning stage

# %%
df.head(len(df))

# %% [markdown]
# ### Removing Unnecessary Features
# Discarding some unused feature columns

# %%
# Drop unnecessary columns and duplicates
cleaned_df = df.drop(columns=['location', 'age', 'year_of_publication', 'publisher', 'Language', 'city', 'state', 'country'])
cleaned_df = cleaned_df.drop_duplicates(subset=['book_title'])

# %% [markdown]
# ### Reduce Categories

# %% [markdown]
# Filtering categories that have less than 50 and more than 2000 books to save computation time.

# %%
# Calculate category counts
category_counts = cleaned_df['Category'].value_counts()

# Filter categories based on count for shorten computing time
unused_cat = category_counts[(category_counts < 50) | (category_counts > 2000)].index.tolist()

# %% [markdown]
# Removes books that have a rating of 0 in the dataset.

# %%
dfbooks = cleaned_df.loc[~cleaned_df['Category'].isin(unused_cat)]
dfbooks = dfbooks[dfbooks['rating'] != 0]
dfbooks.head(len(dfbooks))

# %% [markdown]
# ### Clean up the Category Feature

# %% [markdown]
# This code below will clean the categories feature by removing the symbols and special characters.

# %%
def clean_category(text):
    # Remove square brackets, single/double quotes, and periods
    text = re.sub(r'[\[\]\'"\.]', '', text)
    return text.strip()  # Strip whitespace from both ends of the cleaned text

# Apply the clean_category function to the 'Category' column
dfbooks['category'] = dfbooks['Category'].apply(clean_category)

# Sort unique categories alphabetically
clean_cat_sort = np.sort(dfbooks['category'].unique())

# Print sorted categories
for cat in clean_cat_sort:
    print(cat)

# %% [markdown]
# Drop the 'Category' column

# %%
clean_data = dfbooks.drop(['Category'], axis=1)
clean_data.head()

# %%
clean_data.shape

# %% [markdown]
# ## Modeling Section
# 
# For modeling, content-based filtering and cosine similarity are used to find similarities between books.

# %% [markdown]
# ## Content Based Filtering
# Content Based Filtering is done by creating an algorithm to find out recommendations based on the user's _history_.

# %% [markdown]
# ### TF-IDF Vectorizer

# %% [markdown]
# Used `Tfidfvectorizer` to perform _idf_ calculation on `clean_category` and perform array mapping.

# %%
# Initialize TfidfVectorizer
tf = TfidfVectorizer()
tf.fit(clean_data['category'])
tf.get_feature_names_out()

# %%
# Fit and transform 'clean_category' to TF-IDF matrix
tfidf_matrix = tf.fit_transform(clean_data['category'])
tfidf_matrix.shape

# %% [markdown]
# The result of the previously created matrix can be seen with `todense()`

# %%
tfidf_matrix.todense()

# %%
show_books_category = pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names_out(),
    index=clean_data.book_title
).sample(20, axis=1).sample(10, axis=0)

show_books_category

# %% [markdown]
# ### Cosine Similarity

# %%
# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Create DataFrame with cosine similarity matrix
cosine_sim_df = pd.DataFrame(cosine_sim, index=clean_data['book_title'], columns=clean_data['book_title'])
print(f"Shape: {cosine_sim_df.shape}") 

# Display a sample subset of the similarity matrix
sample_titles = cosine_sim_df.sample(5, axis=1).sample(10, axis=0)  # Sample 5 columns and 10 rows
sample_titles

# %% [markdown]
# ## Recommendation Testing
# This section will try to recommend some books

# %%
def book_recommendations(books, similarity_data=cosine_sim_df, items=clean_data[['book_title', 'rating', 'book_author', 'category']], k=10):
    # Convert the similarity data to a numpy array
    similarity_array = similarity_data.loc[:, books].to_numpy()

    # Get the indices of the k most similar books
    index = similarity_array.argpartition(range(-1, -k, -1))
    
    # Get the book titles and their corresponding similarity scores
    closest_indices = index[-1:-(k+2):-1]
    closest_books = similarity_data.columns[closest_indices]
    closest_scores = similarity_array[closest_indices]

    # Create a DataFrame with the closest books and their similarity scores
    recommendations = pd.DataFrame({
        'book_title': closest_books,
        'similarity_score': closest_scores
    })

    # Drop the queried book from the recommendations
    recommendations = recommendations[recommendations['book_title'] != books]

    # Merge with the items DataFrame to get additional information
    recommendations = recommendations.merge(items, on='book_title').head(k)
    
    return recommendations

# %%
book = "Why We Love Dogs: A Bark & Smile Book"
clean_data[clean_data.book_title.eq(book)]

# %%
book_recommendations(book)


