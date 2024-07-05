# Machine Learning Project Report

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
unused_cat = category_counts[(category_counts < 50) | (category_counts > 2000)].index.tolist()

dfbooks = cleaned_df.loc[~cleaned_df['Category'].isin(unused_cat)]
dfbooks = dfbooks[dfbooks['rating'] != 0]
```

<p align="justify">
Category reduction is done because it will be very time-consuming and memory space if more than 900000 data are input in the model directly. From 982278 rows × 14 columns (cleaned dataset) reduced to 15794 rows × 6 columns. Also, the rating 0 is not included in the table that is why from over 900000 records reduced to 15794 recods.
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
In this project, the model is created with a content-based filtering approach and cosine similarity for similarity measure. The following is an explanation of content based filtering and consine similarity among others:

### Content Based Filtering

<p align="justify">
Content-based filtering is a recommendation method that uses attributes or features of items that users like to suggest similar items. The system analyzes the descriptions and characteristics of items that have been highly rated by users and searches for other items with similar features. The following are the advantages and disadvantages of content based filtering including:
</p>

| Advantages                      | Explanation                                                                                                                                                                                                                                                        | Disadvantages                            | Explanation                                                                                                                                                                                                                                                                      |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Orientation-Based Similarity | Cosine similarity measures the cosine of the angle between two vectors, focusing on the direction rather than the magnitude. This makes it useful for comparing documents of different lengths since it emphasizes the orientation of the vectors over their size. | 1. Ignores Magnitude:                    | While ignoring magnitude can be an advantage, it can also be a disadvantage in situations where the magnitude of the vectors carries important information. Cosine similarity does not account for the length or scale of the vectors, which might be relevant in some contexts. |
| 2. Normalization                | Since cosine similarity is based on the angle between vectors, it inherently normalizes the vectors. This means that it is robust to differences in scale, ensuring that the comparison is not affected by the absolute values of the vector components.           | 2. Sensitivity to Sparse Data:           | In cases where the vectors are very sparse (as is common in high-dimensional spaces like text data), cosine similarity might not be as effective because it relies on common non-zero elements. The similarity score might be skewed by the sparsity of the data.                |
| 3. Computational Efficiency:    | The computation of cosine similarity is relatively efficient, involving basic vector operations like dot product and magnitude calculation. This makes it suitable for large datasets.                                                                             | 3. Non-Euclidean Nature:                 | Cosine similarity does not correspond to a proper metric in a Euclidean space. This means it might not be suitable for all types of clustering or machine learning algorithms that assume a Euclidean distance metric.                                                           |
| 4. Textual Applications:        | In text mining and information retrieval, cosine similarity is particularly effective. It is commonly used to compare documents by converting them into term frequency vectors, making it a standard tool in these domains.                                        | 4. Not Always Intuitive:                 | In some cases, the cosine similarity score might not align with intuitive notions of similarity, especially when comparing vectors with few common dimensions or when the vectors have very different magnitudes.                                                                |
| 5. Interpretability:            | The results of cosine similarity are straightforward to interpret, ranging from -1 (completely dissimilar) to 1 (completely similar), with 0 indicating orthogonality (no similarity).                                                                             | 5. Dependency on Feature Representation: | The effectiveness of cosine similarity is heavily dependent on how the data is represented. Poor feature representation can lead to misleading similarity scores, making it crucial to carefully preprocess and transform the data.                                              |

### Cosine Similarity

<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*LfW66-WsYkFqWc4XYJbEJg.png" height=150>
</div>

<p align="justify">
Cosine similarity measures the similarity between two vectors and determines whether they point in the same direction by calculating the cosine angle between the two vectors. The smaller the cosine angle, the greater the cosine similarity value. The following are the advantages and disadvantages of cosine similarity including:
</p>

| Advantages                       | Explanation                                                                                                                                                                                                                                                        | Disadvantages                            | Explanation                                                                                                                                                                                                                                                                      |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Orientation-Based Similarity: | Cosine similarity measures the cosine of the angle between two vectors, focusing on the direction rather than the magnitude. This makes it useful for comparing documents of different lengths since it emphasizes the orientation of the vectors over their size. | 1. Ignores Magnitude:                    | While ignoring magnitude can be an advantage, it can also be a disadvantage in situations where the magnitude of the vectors carries important information. Cosine similarity does not account for the length or scale of the vectors, which might be relevant in some contexts. |
| 2. Normalization:                | Since cosine similarity is based on the angle between vectors, it inherently normalizes the vectors. This means that it is robust to differences in scale, ensuring that the comparison is not affected by the absolute values of the vector components.           | 2. Sensitivity to Sparse Data:           | In cases where the vectors are very sparse (as is common in high-dimensional spaces like text data), cosine similarity might not be as effective because it relies on common non-zero elements. The similarity score might be skewed by the sparsity of the data.                |
| 3. Computational Efficiency:     | The computation of cosine similarity is relatively efficient, involving basic vector operations like dot product and magnitude calculation. This makes it suitable for large datasets.                                                                             | 3. Non-Euclidean Nature:                 | Cosine similarity does not correspond to a proper metric in a Euclidean space. This means it might not be suitable for all types of clustering or machine learning algorithms that assume a Euclidean distance metric.                                                           |
| 4. Textual Applications:         | In text mining and information retrieval, cosine similarity is particularly effective. It is commonly used to compare documents by converting them into term frequency vectors, making it a standard tool in these domains.                                        | 4. Not Always Intuitive:                 | In some cases, the cosine similarity score might not align with intuitive notions of similarity, especially when comparing vectors with few common dimensions or when the vectors have very different magnitudes.                                                                |
| 5. Interpretability:             | The results of cosine similarity are straightforward to interpret, ranging from -1 (completely dissimilar) to 1 (completely similar), with 0 indicating orthogonality (no similarity).                                                                             | 5. Dependency on Feature Representation: | The effectiveness of cosine similarity is heavily dependent on how the data is represented. Poor feature representation can lead to misleading similarity scores, making it crucial to carefully preprocess and transform the data.                                              |

<p align="justify">
In building a book recommendation system model with Content Based Filtering, the first thing to do is to weight the category feature using the <b>TfidfVectorizer</b> module from the sklearn library to get what categories exist. Next, the <b>cosine_similarity</b> module from the sklearn library is used. Then a recommendation system function is created setting the parameter k = 10 which means it will issue recommendations for the top 10 books based on the category and also the items parameter which contains the book title, rating, author used to define similarity.
</p>

### Results

#### Recommendation Testing

<p align="justify">
The following is a code snippet to create a recommendation from the model, where 1 randomly selected book title named 'Why We Love Dogs: A Bark & Smile Book' is a book about pets.
</p>

```python
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
```

From the code above, the following recommendation results are obtained:

| book_title                                                                                                                                           | similarity_score | rating | book_author                 | category |
|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|--------|-----------------------------|----------|
| Pug Shots                                                                                                                                            | 1.0              | 10     | Jim Dratfield               | Pets     |
| Dog Perfect: The User-Friendly Guide to a Well-Behaved Dog                                                                                           | 1.0              | 9      | Sarah Hodgson               | Pets     |
| Aspca Complete Cat Care Manual                                                                                                                       | 1.0              | 10     | Andrew Edney                | Pets     |
| Cosmic Canines : The Complete Astrology Guide for You and Your Dog (Native Agents Series)                                                            | 1.0              | 8      | MARILYN MACGRUDER BARNEWALL | Pets     |
| A Step-By-Step Book About Gerbils                                                                                                                    | 1.0              | 10     | Patrick Bradley             | Pets     |
| The Evans Guide for Housetraining Your Dog                                                                                                           | 1.0              | 9      | Job Michael  Evans          | Pets     |
| The New Parrot Handbook: Everything About Purchase, Acclimation, Care, Diet, Disease, and Behavior Od Parrots, With a Special Chapter on Raising Par | 1.0              | 8      | Werner Lantermann           | Pets     |
| Beyond Basic Dog Training                                                                                                                            | 1.0              | 7      | Diane L. Bauman             | Pets     |
| The Trick Is in the Training: 25 Fun Tricks to Teach Your Dog                                                                                        | 1.0              | 9      | Stephanie J. Taunton        | Pets     |
| The Essential Beagle (Essential (Howell))                                                                                                            | 1.0              | 9      | Howell Book House           | Pets     |


<p align="justify">
From the above results, the recommendation system model can provide recommendations for 10 books according to the parameters used and the similarity of books about pets.
</p>

## Evaluation

<p align="justify">
In modeling the recommendation system, the evaluation metric used is precision. The precision used is to calculate how many relevant items are recommended divided by the number of items that are recommended. The following is the precision formula:
</p>

![pres](https://user-images.githubusercontent.com/57740421/196231953-e943707a-8221-4f64-80f7-d6826a514c58.png)

Based on the previous results, 10 book recommendations were obtained and the relevant items were also 10 so the precision obtained was 10/10 = 1 or 100%. With this precision, the model that has been built is good to be used as a recommendation system with content based filtering.

## References

[1] &emsp; M. R. A. Zayyad, “Sistem Rekomendasi Buku Menggunakan Metode Content Based Filtering,” dspace.uii.ac.id, Jul. 2021, Accessed: Jun. 26, 2024. [Online]. Available: https://dspace.uii.ac.id/handle/123456789/35942
