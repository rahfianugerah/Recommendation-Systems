# Machine Learning Project Report - Naufal Rahfi Anugerah

## Project Domain
<p align="justify">
In this fast-paced digital age, the number of books available online and offline is increasing at a rapid pace. Books are a source of information and knowledge to broaden the horizons in various ways. With more and more information about different types of books, Finding books that suit individual interests and preferences can be a very challenging task [1]. With the recommendation system, it is hoped that it can facilitate readers in getting the book they want and can shorten the time in searching for books. 
</p>

## Business Understanding

<p align="justify">
In the process of clarifying the problem, a problem statements have been identified as primary focuses, along with establishing goals to be achieved for these problem statements.
</p>

### Problem Statements

<p align="justify">
Based on the background above, the problem is how can we design and develop an precise and efficient book recommendation system, which is able to understand user preferences and provide relevant and personalized recommendations, thus helping users find the right book without having to spend a lot of time searching manually?
</p>

### Goals

<p align="justify">
The goal is to build a book recommendation system model that can be used to recommend books according to the content preferred by readers.
</p>

### Solution statements

<p align="justify">
To create a book recommendation system model with a content-based filtering approach and measure the similarity of books to be recommended using consine similarty. Also, evaluate the model with a precision evaluation metric to measure the accuracy of the model in providing book recommendations.
</p>

## Data Understanding
In making this project, the dataset used is a dataset taken from
- Kaggle: [Click here!](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset/data)

This project only uses 1 dataset from the link listed above and th dataset contains 19 feature in the *Preprocessed_data.csv*. The dataset contains 1031175 rows × 19 columns

### Dataset Variables
- user_id: id of a user
- location: location of the readers
- age: age of the readers
- isbn: book identification codes
- rating: book rating from readers
- book_title: title of the book
- book_author: book author name
- year_of_publication: merupakan tahun publikasi buku
- publisher: merupakan penerbit buku
- img_s: book cover link (size small)
- img_m: book cover link (size medium)
- img_l: book cover link (size large)
- Summary: book synopsis
- Language: book languages
- category: book categories
- city: the city where the book was purchased
- state: the state where the book was purchased
- country: the country where the book was purchased

### Visualization & Exploratory Data Analysis
<p align="justify">
To understand the information in the dataset used, visualization and data analysis stages are carried out that can provide insight or new information. The following are some data visualizations including:
</p>

#### Rating Distribution

<div align="center">
    <img src="https://github.com/rahfianugerah/recommendation-system/blob/main/img/rating.png?raw=true" height=400>
</div>

<p align=justify>
Based on the results of the visualization above, information can be taken in the form of many books that are still rated 0. Most likely there are many new books that have not been rated or it could be many books that not many people know about.
</p>

#### Cleaned Rating Distribution

<div align="center">
    <img src="https://github.com/rahfianugerah/recommendation-system/blob/main/img/ratingclean.png?raw=true" height=400>
</div>

<p align=justify>
Based on the results of the visualization above, information can be taken in the form of many books that are rated 8 out of 10 in the first rank with more than 80000 books while books with a score of 10 out of 10 are in the second rank. It can be concluded that books with a score of 8 out of 10 are more widely read and more popular among readers than books with a score of 10 out of 10.
</p>


#### Top 10 Most Readed Books
<div align="center">
    <img src="https://github.com/rahfianugerah/recommendation-system/blob/main/img/books.png?raw=true" height=400>
</div>

<p align=justify>
Based on the results of the visualization above, information can be taken in the form of a book with the title wild animus has been read by more than 2000 readers, which means that this book is very popular among readers.
</p>

#### Top 10 Book Authors
<div align="center">
    <img src="https://github.com/rahfianugerah/recommendation-system/blob/main/img/author.png?raw=true" height=400>
</div>

<p align=justify>
Based on the results of the visualization above, information can be drawn that Stepehen King is a best-selling author. Where many people have read books that have been written by him.
</p>

#### Top 10 Years of Publication
<div align="center">
    <img src="https://github.com/rahfianugerah/recommendation-system/blob/main/img/publication.png?raw=true" height=400>
</div>

<p align=justify>
Based on the results of the visualization above, information can be taken in 2002 as the year with the most book publications. Which means that many books have been published in 2002.
</p>

#### Top 10 Book Publisher
<div align="center">
    <img src="https://github.com/rahfianugerah/recommendation-system/blob/main/img/publisher.png?raw=true" height=400>
</div>

<p align=justify>
Based on the visualization results above, information can be taken in the form of book publishers that publish the most books is Ballantine Books with more than 30000 books. This shows that Ballantine Books has a large contribution in the publishing industry during the specific time period observed.
</p>

## Data Preparation

### Removing Unnecessary Features

<p align="justify">
In this section, we will clean up features that are not used in making recommendation systems. This is done so that the model can compute efficiently. The following are the code snippets used:
</p>

```python
# Drop unnecessary columns and duplicates
cleaned_df = df.drop(columns=['location', 'age', 'year_of_publication', 'Summary', 'Language', 'city', 'state', 'country'])
cleaned_df = cleaned_df.drop_duplicates(subset=['book_title'])
```

### Reduce Categories

<p align="justify">
In this section, category reduction is done so that the data can be accepted by the model to work more efficiently and reduce computation time due to very large data. The following are the code snippets used:
</p>

```python
# Calculate category counts
category_counts = cleaned_df['Category'].value_counts()
# Filter categories based on count for shorten computing time
unused_cat = category_counts[(category_counts < 100) | (category_counts > 1000)].index.tolist()

dfbooks = cleaned_df.loc[~cleaned_df['Category'].isin(unused_cat)]
dfbooks = dfbooks[dfbooks['rating'] != 0]
```

<p align="justify">
Category reduction is done because it will be very time-consuming and memory space if more than 900000 data are input in the model directly. From 982278 rows × 14 columns (cleaned dataset) reduced to 9076 x 6 columns. Also, the rating 0 is not included in the table that is why from over 900000 records reduced to 9000 recods.
</p>

### Clean up the Category Feature
<p align="justify">
In this section, the category features will be cleaned before being processed into the model, here is a table where the category features have not been cleaned:
</p>

| isbn      | rating | book_tittle                                       | book_author         | publisher                   | Category               |
|------------|--------|---------------------------------------------------|---------------------|-----------------------------|------------------------|
| 157663937  | 6      | More Cunning Than Man: A Social History of Rat... | Robert Hendrickson  | Kensington Publishing Corp. | ['Nature']             |
| 1879384493 | 10     | If I'd Known Then What I Know Now: Why Not Lea... | J. R. Parrish       |               Cypress House | ['Reference']          |
| 0375509038 | 8      | The Right Man : The Surprise Presidency of Geo... | DAVID FRUM          | Random House                | ['Political Science']  |
| 8476409419 | 8      | Estudios sobre el amor                            | Jose Ortega Y Gaset |        Downtown Book Center | ['Literary Criticism'] |
| 3498020862 | 8      |                                  Die Korrekturen. | Jonathan Franzen    |            Rowohlt, Reinbek | ['American fiction']   |

<p align="justify">
By using the regular expression library, the category feature can be cleaned up with the code snippet below:
</p>

```python
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
```
<p align="justify">
After cleaning, the remaining category features are just strings without any symbols or special characters and can be used for the modeling stage.
</p>


| isbn      | rating | book_tittle                                       | book_author         | publisher                   | Category               |
|------------|--------|---------------------------------------------------|---------------------|-----------------------------|------------------------|
| 157663937  | 6      | More Cunning Than Man: A Social History of Rat... | Robert Hendrickson  | Kensington Publishing Corp. | Nature             |
| 1879384493 | 10     | If I'd Known Then What I Know Now: Why Not Lea... | J. R. Parrish       |               Cypress House | Reference          |
| 0375509038 | 8      | The Right Man : The Surprise Presidency of Geo... | DAVID FRUM          | Random House                | Political Science  |
| 8476409419 | 8      | Estudios sobre el amor                            | Jose Ortega Y Gaset |        Downtown Book Center | Literary Criticism |
| 3498020862 | 8      |                                  Die Korrekturen. | Jonathan Franzen    |            Rowohlt, Reinbek | American fiction   |

## Modeling and Result
In this project, the model is created with a content-based filtering approach and Cosine similarity for similarity measure. The following is an explanation of content based filtering and consine similarity among others:

### Content Based Filtering

<p align="justify">
Content-based filtering is a recommendation method that uses attributes or features of items that users like to suggest similar items. The system analyzes the descriptions and characteristics of items that have been highly rated by users and searches for other items with similar features. The following are the advantages and disadvantages of content based filtering including:
</p>

#### Advantages of Content-Based Filtering

<p align="justify">
1. Personalization: 

Content-based filtering provides personalized recommendations based on the user's preferences and past interactions with items. This can lead to a more satisfying user experience as recommendations are tailored to individual tastes.
</p>

<p align="justify">
2. Transparency: 

The recommendations are based on explicit features or attributes of items (such as genres, keywords, or content descriptions), making it transparent to understand why certain items are recommended.
</p>

<p align="justify">
3.No Cold Start Problem: 

Content-based filtering can recommend items to new users because it does not require historical data about other users. As long as the system has enough information about the item itself, it can make recommendations.
</p>

<p align="justify">
4. Reduced Dependence on Data: 

Content-based filtering can work effectively even with sparse user data because it primarily relies on item features. This makes it suitable for scenarios where user data collection is limited or privacy concerns restrict data availability.
</p>

<p align="justify">
5. Serendipity: 

While primarily focusing on similar items, content-based filtering can also introduce diversity in recommendations by recommending items with overlapping attributes that the user might not have encountered otherwise.
</p>

#### Disadvantages of Content-Based Filtering

<p align="justify">
1. Limited Serendipity: 

Content-based filtering tends to recommend items similar to those already liked or interacted with by the user. This can create a filter bubble where users are not exposed to new or diverse items outside their established preferences.
</p>

<p align="justify">
2. Over-Specialization:

If the item features used for recommendations are too specific or narrow, it can lead to over-specialization and recommendations that do not capture the user's broader interests or evolving preferences.
</p>

<p align="justify">
3. Scalability: 

Building and maintaining content-based recommendation systems can be resource-intensive, especially if the item catalog is large or if detailed feature extraction is computationally expensive.
</p>

<p align="justify">
4. Cold Start for New Items: 

Content-based filtering struggles with recommending newly added items or items with limited feature information, as it relies heavily on existing item profiles.
</p>

<p align="justify">
5.Dependency on Feature Extraction: 

The effectiveness of content-based filtering heavily relies on the quality and relevance of the features or attributes used to describe items. If these features are not well-defined or extracted inaccurately, the quality of recommendations can suffer.
</p>

### Cosine Similarity

<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*LfW66-WsYkFqWc4XYJbEJg.png" height=150>
</div>

<p align="justify">
Cosine similarity measures the similarity between two vectors and determines whether they point in the same direction by calculating the cosine angle between the two vectors. The smaller the cosine angle, the greater the cosine similarity value. The following are the advantages and disadvantages of cosine similarity including:
</p>

#### Advantages of Cosine Similarity

<p align="justify">
1. Orientation-Based Similarity:

Cosine similarity measures the cosine of the angle between two vectors, focusing on the direction rather than the magnitude. This makes it useful for comparing documents of different lengths since it emphasizes the orientation of the vectors over their size.
</p>

<p align="justify">
2. Normalization:

Since cosine similarity is based on the angle between vectors, it inherently normalizes the vectors. This means that it is robust to differences in scale, ensuring that the comparison is not affected by the absolute values of the vector components.
</p>

<p align="justify">
3. Computational Efficiency:

The computation of cosine similarity is relatively efficient, involving basic vector operations like dot product and magnitude calculation. This makes it suitable for large datasets.
</p>

<p align="justify">
4. Textual Applications:

In text mining and information retrieval, cosine similarity is particularly effective. It is commonly used to compare documents by converting them into term frequency vectors, making it a standard tool in these domains.
</p>

<p align="justify">
5. Interpretability:

The results of cosine similarity are straightforward to interpret, ranging from -1 (completely dissimilar) to 1 (completely similar), with 0 indicating orthogonality (no similarity).
</p>

#### Disadvantages of Cosine Similarity

<p align="justify">
1. Ignores Magnitude:

While ignoring magnitude can be an advantage, it can also be a disadvantage in situations where the magnitude of the vectors carries important information. Cosine similarity does not account for the length or scale of the vectors, which might be relevant in some contexts.
</p>

<p align="justify">
2. Sensitivity to Sparse Data:

In cases where the vectors are very sparse (as is common in high-dimensional spaces like text data), cosine similarity might not be as effective because it relies on common non-zero elements. The similarity score might be skewed by the sparsity of the data.
</p>

<p align="justify">
3. Non-Euclidean Nature:

Cosine similarity does not correspond to a proper metric in a Euclidean space. This means it might not be suitable for all types of clustering or machine learning algorithms that assume a Euclidean distance metric.
</p>

<p align="justify">
4. Not Always Intuitive:

In some cases, the cosine similarity score might not align with intuitive notions of similarity, especially when comparing vectors with few common dimensions or when the vectors have very different magnitudes.
</p>

<p align="justify">
5. Dependency on Feature Representation:

The effectiveness of cosine similarity is heavily dependent on how the data is represented. Poor feature representation can lead to misleading similarity scores, making it crucial to carefully preprocess and transform the data.
</p>


### Results

#### Wordcloud

<div align="center">
    <img src="https://github.com/rahfianugerah/recommendation-system/blob/main/img/wordcloud.png?raw=true" height=250>
</div>

<p align="justify">
From the figure above, the book categories 'Literary Criticism', 'Sport Recreation', 'Poetry', 'Travel', 'Performing Arts', 'Comics Graphic' are the 6 most popular book categories based on the dataset used.
</p>

#### Cosine Similarity Measure

<p align="justify">
The following is a code snippet which searches for the same correlation with the input book using the results of the consine similarity measure.
</p>

```python
# Function to get top n similar books for a given book
def get_similar_books(book_title, n=5):
    similar_books = cosine_sim_df[book_title].sort_values(ascending=False).head(n+1).iloc[1:]
    return similar_books

# Example: Get top 5 books similar to a specific book
book_title = 'Physics'  # Replace with an actual book title from your dataset
similar_books = get_similar_books(book_title)
print(f"Top 5 books similar to '{book_title}':\n{similar_books}")
```
Result:
```
Top 5 books similar to 'Physics':
book_title
The Strange Story of the Quantum                                                1.0
The World According To Pimm: A Scientist Audits the Earth                       1.0
X-Raying the Pharaohs                                                           1.0
The Science of Jurassic Park: And the Lost World Or, How to Build a Dinosaur    1.0
Rising From The Plains                                                          1.0
Name: Physics, dtype: float64
```

<p align="justify">
From the code above, after 1 random book title is entered into the code, 5 random books with perfect correlation or value 1 are obtained using consine similarity.
</p>

#### Recommendation Testing

<p align="justify">
The following is a code snippet to create a recommendation from the model, where 1 randomly selected book title named 'Die Korrekturen' with the book category 'American fiction' will be tried.
</p>

```python
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
```

<p align="justify">
The following is the result of the recommendation system model, where k = 10, and produces 10 book recommendations based on the highest rating to the predetermined low rating limit, namely books with a rating of 5.
</p>

| isbbn      | rating | book_tittle                                       | book_author                        | publisher                   | clean_category   |
|------------|--------|---------------------------------------------------|------------------------------------|-----------------------------|------------------|
| 156731600X | 10     |                              On Writers & Writing | John Gardner                       | MJF Books                   | American fiction |
| 0553148001 | 9      |               The Clan of the Cave Bear : a novel | Jean M. Auel                       |                Bantam Books | American fiction |
| 0762101067 | 8      |                          Best of Sisters In Crime | Marilyn Wallace                    | Penguin Putnam~childrens Hc | American fiction |
| 2868691870 | 8      | CitÃ?Â© de verre                                  | Paul Auster                        |                   Actes Sud | American fiction |
| 1841952338 | 7      |           Happiness TM (Building Business Skills) | Will Ferguson                      |         Canongate Books Ltd | American fiction |
| 0451159535 | 7      |                               Coma (Signet Books) | Robin Cook                         | Signet Book                 | American fiction |
| 2253171689 | 7      | Toxine                                            | Robin Cook                         | Le Livre de Poche           | American fiction |
| 0871131447 | 5      |                                     Not Fade Away | Jim Dodge                          |              Pub Group West | American fiction |
| 3453131150 | 5      | Der weite Himmel.                                 | Nora Roberts                       | Heyne                       | American fiction |
| 0898151279 | 5      | Ladies' Own Erotica: Tales, Recipes, and Other... | Kensington Ladies' Erotica Society | Ten Speed Press             | American fiction |

## Evaluation

<p align="justify">
In modeling the recommendation system, the evaluation metric used is precision. Precision is a measure of how many results retrieved by the system are actually relevant out of the total results retrieved. It is a metric used to evaluate the accuracy of the results deemed relevant by the system. The following is the precision formula:
</p>

$$TP/TP+FP$$

- $TP$ (True Positive): The number of instances correctly predicted as positive.
- $FP$ (False Positive): The number of instances that were incorrectly predicted as positive.

<p align="justify">
This project will not use True positive or False Positive but the rating given to the book to determine whether the recommended book is relevant or not. The following is the form of the formula used:
</p>

$$Precision@k= Relevant Recommended Items/ItemsRecommended$$

Relevant there refers to the following conditions:
- If the rating >= 5 is referred to as relevant
- If the rating < 5 is referred to as irrelevant



The following is a code snippet to evaluate the recommendation system using the precision metric:

```python
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
```
Result:

```
Precision at K: 1.00
```

<p align="justify">
From the snippet above, a precision of 1 or 100% is obtained, this means that all results taken by the model are truly relevant. In other words, every instance classified as positive by the model is indeed positive. 
</p>

<p align="justify">
The following are some of the impacts obtained after testing the results of the book recommendation system including:
</p>

- High Precision: With 100% precision, the model only recommends relevant and targeted products to users. This is very positive as users will not receive irrelevant recommendations, which can improve user satisfaction and experience.
- User Trust: High precision also means that users can have more trust in the recommendation system as every recommendation given is truly in line with their interests or needs.



## References

[1] &emsp; M. R. A. Zayyad, “Sistem Rekomendasi Buku Menggunakan Metode Content Based Filtering,” dspace.uii.ac.id, Jul. 2021, Accessed: Jun. 26, 2024. [Online]. Available: https://dspace.uii.ac.id/handle/123456789/35942
