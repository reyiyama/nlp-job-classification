# Job Ads Classification Using Natural Language Processing

## Project Overview
This project aims to build an automated classification system for job advertisements to improve the efficiency of job posting categorization on job search websites. It involves natural language processing (NLP) techniques for data preprocessing, feature extraction, and machine learning-based classification of job ads into specific categories like Accounting/Finance, Engineering, Healthcare/Nursing, and Sales. This helps reduce manual errors in category assignment, ensuring job posts are accurately classified and therefore improving the visibility to relevant job seekers.

The project is divided into two milestones:
1. **Milestone 1:** Preprocess the text data, generate feature representations, and classify job ads into different categories using machine learning models.
2. **Milestone 2:** (Not included here) Utilize the classification model to build a job-hunting website.

This README covers **Milestone 1**, where we used Python and various NLP techniques to preprocess job ads data, represent them as feature vectors, and classify them.

## Project Motivation
Job portals often face challenges where job categories are incorrectly assigned, resulting in reduced visibility to relevant candidates. By automating the classification of job ads using NLP and machine learning, job portals can:
- Minimize human errors in data entry.
- Improve the exposure of job ads to the most relevant audience.
- Enhance the overall user experience for both job seekers and advertisers.

## Steps and Methods

### 1. Basic Text Preprocessing
The preprocessing steps included:
- **Tokenization:** Splitting the job ad descriptions into individual tokens (words) using the regular expression: `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`.
- **Lowercase Conversion:** All words were converted to lowercase.
- **Removing Short Words:** Words with fewer than 2 characters were removed.
- **Stop Words Removal:** A custom stopword list (`stopwords_en.txt`) was used to remove common words that do not contribute to the meaning.
- **Rare and Frequent Words Removal:** Words that appeared only once were removed, along with the top 50 most frequent words based on document frequency.
- **Building Vocabulary:** A vocabulary of the cleaned text was generated and saved in a file called `vocab.txt`, where words were sorted in alphabetical order, starting with an index of 0.

The processed data was saved, including a vocabulary file:
- **vocab.txt** contains entries in the format `word:index`.

### 2. Generating Feature Representations
To represent the job ads text data as numerical vectors suitable for machine learning, we generated three types of feature representations:
- **Count Vector (Bag-of-Words):** A sparse representation of word frequencies in each job ad, saved in `count_vectors.txt`. This representation only considered the words present in the `vocab.txt` file.
- **Word Embeddings:** We used FastText pre-trained embeddings to represent each job ad.
  - **Unweighted Embeddings:** Averaging word vectors to represent the entire document.
  - **TF-IDF Weighted Embeddings:** Weighting word vectors by their term frequency-inverse document frequency (TF-IDF) values before averaging.

### 3. Job Advertisement Classification
To classify the job ads into the correct categories, we built and evaluated machine learning models.
- **Logistic Regression:** We used logistic regression as a simple yet effective baseline model for classification. The model was trained using the different feature representations generated in Task 2.
- **Cross-Validation:** A 5-fold cross-validation approach was employed to ensure robust performance evaluation.
- **Experiments:**
  - **Model Comparison:** We compared the performance of models using different feature representations (Count Vector, Unweighted Embeddings, TF-IDF Weighted Embeddings).
  - **Effect of Extra Information:** We also experimented with adding job titles to descriptions to see if more information helped improve the model's accuracy.

## Results
- **Feature Representation Performance:** TF-IDF Weighted Embeddings provided the best performance in terms of accuracy, followed by unweighted embeddings and count vector representation.
- **Impact of Additional Information:** Including both the job title and description yielded a slight improvement in classification accuracy compared to using the description alone.

## Libraries Used
- **pandas, numpy, re** for data manipulation and regular expressions.
- **nltk** for natural language processing tasks (tokenization).
- **gensim** for loading pre-trained FastText embeddings.
- **sklearn** for machine learning modeling and evaluation.
- **Jupyter Notebook** for an interactive coding environment.

## Reproducing the Results
To replicate the results and findings from the project, follow the steps below:

### Prerequisites
1. Install Python 3.9 or higher.
2. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
3. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn nltk gensim
   ```
Below, I’ve provided a detailed explanation of what was done in the `task1.ipynb` and `task2_3.ipynb` notebooks, along with instructions for recreating the findings using the attached `.txt` files.

### Data Setup
1. **Download Data:** Ensure you have the 'data' folder that contains the subfolders for each job category (Accounting_Finance, Engineering, Healthcare_Nursing, Sales).
2. **Stopwords File:** Place `stopwords_en.txt` in the same folder as the data.

### Summary of `task1.ipynb` - Basic Text Preprocessing
This notebook focused on extracting, cleaning, and tokenizing job advertisement data to prepare it for further analysis. Here’s what was done:

1. **Examining and Loading Data**:
   - The job advertisements were organized in folders by categories such as Accounting_Finance, Engineering, Healthcare_Nursing, and Sales.
   - Each job advertisement (`Job_<ID>.txt`) was loaded, and key data was extracted into a DataFrame with columns for the job category, job ID, and content.

2. **Preprocessing Steps**:
   - **Tokenization**: The job descriptions were tokenized using a regular expression: `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`. This splits the description into words while preserving hyphens and apostrophes.
   - **Lowercase Conversion**: All words were converted to lowercase for consistency.
   - **Removing Short Words**: Words shorter than two characters were removed.
   - **Stop Words Removal**: A custom stopword list (`stopwords_en.txt`) was used to remove common words such as “the,” “is,” etc.
   - **Removing Rare Words**: Words that appeared only once across the entire dataset were removed to reduce noise.
   - **Removing Top 50 Frequent Words**: The top 50 most frequent words were also removed to improve the model’s ability to differentiate between job ads.
   - **Building Vocabulary**: A vocabulary file (`vocab.txt`) was generated from the processed tokens, listing unique words and assigning an index to each. The words were sorted alphabetically and saved in the format `word:index`.

3. **Saving Outputs**:
   - The cleaned content of each job advertisement was saved in a new column named `Processed_Content`.
   - The vocabulary was saved to `vocab.txt`.

#### Instructions to Recreate Findings for `task1.ipynb`
1. **Run Task 1 Notebook**:
   - Ensure the data folder structure is intact (subfolders for each job category with `.txt` files).
   - Load the data into the notebook and process it as described above.
   - Generate `vocab.txt` by running the pre-processing cells in the notebook.

### Summary of `task2_3.ipynb` - Generating Feature Representations and Classification
This notebook focused on generating feature representations and building machine learning models to classify job ads into categories. 'vocab.txt' is saved in the directory.

1. **Generating Feature Representations**:
   - **Bag-of-Words Representation**: The processed job descriptions were represented as sparse count vectors based on the vocabulary generated in Task 1 (`vocab.txt`). Each job ad was represented by a set of word counts, which were saved in `count_vectors.txt`.
   - **Word Embeddings**:
     - **Unweighted Embeddings**: FastText pre-trained embeddings were used to create word vectors for each job ad. The average word embedding was calculated for each document.
     - **TF-IDF Weighted Embeddings**: TF-IDF weighting was applied to the FastText embeddings to give more importance to terms that are highly relevant to each job ad.

2. **Job Advertisement Classification**:
![image](https://github.com/user-attachments/assets/a68ea55e-ab3a-4aca-a05d-5fd1ee8d1e1b)

![image](https://github.com/user-attachments/assets/84abe494-110a-4fc3-97ad-e71237e82d59)

![image](https://github.com/user-attachments/assets/7f0bbc36-499f-46cc-8bcc-9cf4f8620832)



   - **Models Built**: Logistic Regression was used as the primary model to classify job ads.
   - **Cross-Validation**: A 5-fold cross-validation was performed to evaluate model performance.
   - **Experiments**:
     - **Model Comparison**: The accuracy of different feature representations (Count Vector, Unweighted Embeddings, TF-IDF Weighted Embeddings) was compared.
     - **Additional Features**: The impact of including job titles in addition to descriptions was also investigated.
   - **Results Summary**:
     - The **TF-IDF Weighted Embeddings** provided the best classification accuracy, followed by unweighted embeddings and Bag-of-Words representations.
     - **Adding Job Titles** yielded only a slight improvement in model performance compared to using job descriptions alone.

#### Instructions to Recreate Findings for `task2_3.ipynb`
1. **Run Feature Representation Notebook**:
   - Ensure `vocab.txt` is available in the working directory.
   - Use the provided data (`count_vectors.txt` and the cleaned content from Task 1).
   - Run the cells to generate feature representations:
     - **Count Vectors**: Use `vocab.txt` to create the count vectors and save them in `count_vectors.txt`.
     - **Word Embeddings**: Load the FastText model, and use the tokenized descriptions to compute average embeddings.
     - **TF-IDF Weighted Embeddings**: Use the TF-IDF vectorizer to compute the embeddings.

2. **Run Classification Models**:
   - **Bag-of-Words**: Load the count vectors (`count_vectors.txt`) and split them into training and testing sets.
   - **Embedding Models**: Use the embeddings generated for training the classification models.
   - Perform **5-fold cross-validation** to evaluate the performance of each representation.

### Results Summary for `task2_3.ipynb`
- The **Bag-of-Words with Multinomial Naive Bayes** provided the highest accuracy (0.8698) for job descriptions.
- Adding **job titles** to descriptions did not always lead to significant performance improvements, but in some cases, it slightly increased accuracy.
- **TF-IDF Weighted Embeddings** improved over plain embeddings, especially when used with Gradient Boosting Machines and Random Forest classifiers.

![image](https://github.com/user-attachments/assets/4475966e-4bbe-490b-a6d7-f19dbafc0f0a)

By following these detailed instructions, you should be able to recreate the findings and outputs of `task1.ipynb` and `task2_3.ipynb` accurately.

### Running the Notebooks
1. **Task 1 (Text Preprocessing):** Run the `task1.ipynb` notebook to perform data preprocessing and generate `vocab.txt`.
2. **Task 2 & 3 (Feature Representation and Classification):** Run the `task2_3.ipynb` notebook to generate feature representations and build classification models. This will produce `count_vectors.txt` and other outputs.

### Expected Output Files
- **vocab.txt:** Contains vocabulary terms indexed.
- **count_vectors.txt:** Stores the count vector representation of each job ad.

## Future Work
- **Milestone 2:** Integrate the classification model into a web application to provide an intuitive interface for job seekers and employers.
- **Advanced Models:** Explore the use of transformer-based models like BERT to further enhance classification accuracy.

## Industry Importance
Automated job ad classification can significantly enhance the relevance of job searches, reducing the manual effort required for job categorization, and ultimately helping job seekers find the most suitable opportunities efficiently. The methods applied in this project are directly applicable to any job portal or HR system aiming to improve its data processing capabilities and user satisfaction.

## Author
- **Amay Viswanathan Iyer** (Student ID: 3970066)

