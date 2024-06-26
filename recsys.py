# %% [markdown]
# # Machine Learning Project - Book Recomendation Systems
# 
# - Name: Naufal Rahfi Anugerah
# - Dicoding ID: nrahfi
# - E-mail: 189nrahfi@gmail.com

# %% [markdown]
# ## Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import zipfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Data Gathering

# %% [markdown]
# ### Download Dataset

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

# %%
df = pd.read_csv("Books Data with Category Language and Summary\Preprocessed_data.csv")
df.head(len(df))

# %% [markdown]
# ## Data Assesing

# %%
df.info()

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

# %%
df = pd.DataFrame(df.dropna())
df.head(len(df))

# %%
data_assesing(df)

# %%
df = pd.DataFrame(df.drop(columns=['img_s', 'img_l', 'img_m', 'Unnamed: 0', 'user_id']))

# %%
df.head()

# %% [markdown]
# ## Exploratory Data Analysis

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

# %%
# Count occurrences of each book and select top 10
data_authors = df['book_title'].value_counts().head(10).reset_index()
data_authors.columns = ['book_title', 'count']

# Plotting using seaborn and matplotlib
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='book_title', data=data_authors, color='skyblue')

# Customizing labels and title
plt.xlabel('Count', size=10)
plt.ylabel('Author', size=10)
plt.title('Top 10 Books', size=10)

# Adjusting tick label size for better readability
plt.xticks(size=10)
plt.yticks(size=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid lines with transparency (alpha=0.7)

# Displaying the plot
plt.show()

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
# ## Data Preparation

# %%
df.head(len(df))

# %%
# Drop unnecessary columns and duplicates
cleaned_df = df.drop(columns=['location', 'age', 'year_of_publication', 'Summary', 'Language', 'city', 'state', 'country'])
cleaned_df = cleaned_df.drop_duplicates(subset=['book_title'])

# Calculate category counts
category_counts = cleaned_df['Category'].value_counts()

# Filter categories based on count for shorten computing time
unused_cat = category_counts[(category_counts < 100) | (category_counts > 1000)].index.tolist()

# %%
dfbooks = cleaned_df.loc[~cleaned_df['Category'].isin(unused_cat)]
dfbooks = dfbooks[dfbooks['rating'] != 0]
dfbooks.head(len(dfbooks))

# %%
def clean_category(text):
    # Remove square brackets, single/double quotes, and periods
    text = re.sub(r'[\[\]\'"\.]', '', text)
    return text.strip()  # Strip whitespace from both ends of the cleaned text

# Apply the clean_category function to the 'Category' column
dfbooks['clean_category'] = dfbooks['Category'].apply(clean_category)

# Sort unique categories alphabetically
clean_cat_sort = np.sort(dfbooks['clean_category'].unique())

# Print sorted categories
for cat in clean_cat_sort:
    print(cat)

# %%
clean_data = dfbooks.drop(['Category'], axis=1)
clean_data.head()

# %% [markdown]
# ## Modeling Section

# %% [markdown]
# ### TF-IDF Vectorizer

# %%
# Initialize TfidfVectorizer
tf = TfidfVectorizer()

# Fit and transform 'clean_category' to TF-IDF matrix
tfidf_matrix = tf.fit_transform(clean_data['clean_category'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# %% [markdown]
# ### Cosine Similarity

# %%
# Create DataFrame with cosine similarity matrix
cosine_sim_df = pd.DataFrame(cosine_sim, index=clean_data['book_title'], columns=clean_data['book_title'])

# Display a sample subset of the similarity matrix
sample_titles = cosine_sim_df.sample(5, axis=1).sample(10, axis=0)  # Sample 5 columns and 10 rows
sample_titles

# %% [markdown]
# ## Recommendation Testing

# %%
def get_recommendations(book_title):
    # Get the index of the book_title in cosine_sim_df
    index = cosine_sim_df.index.get_loc(book_title)

    # Get k most similar books (excluding the book_title itself)
    k = 10
    closest_indices = np.argsort(cosine_sim_df.iloc[index].values)[-2:-(2 + k):-1]
    closest_books = cosine_sim_df.columns[closest_indices]

    # Filter items DataFrame to include only recommended books with rating > 5
    recommended_books = clean_data[(clean_data['book_title'].isin(closest_books)) & (clean_data['rating'] >= 5)]

    return recommended_books.sort_values(by='rating', ascending=False).head(k)

# Example usage:
books = 'Die Korrekturen.'
recommended_books = get_recommendations(books)
recommended_books.sort_values(['rating'], ascending=True)
pd.DataFrame(recommended_books)

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### Precision

# %%
def precision_at_k(recommended_books, ground_truth_books):
    """
    Calculate the precision at k for the recommended books.
    
    Parameters:
    - recommended_books: DataFrame, books recommended by the system
    - ground_truth_books: DataFrame, ground truth relevant books
    
    Returns:
    - float, precision at k
    """
    # Calculate the number of relevant items in the recommended books
    relevant_items = recommended_books[recommended_books['book_title'].isin(ground_truth_books['book_title'])]

    # Precision is the number of relevant items retrieved divided by the number of items retrieved
    precision = len(relevant_items) / len(recommended_books) if len(recommended_books) > 0 else 0
    return precision

# Define a threshold
threshold = 5

# Define a ground truth DataFrame for evaluation
ground_truth_books = clean_data[clean_data['rating'] >= threshold]

# Calculate precision at k
precision = precision_at_k(recommended_books, ground_truth_books)
print(f"Precision at K: {precision:.2f}")


