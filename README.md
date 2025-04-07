# Analysis of Bias in Recruitment Systems Utilizing Large Language Models

## Authors

- Nupur Deshmukh 
- Sharayu Sunil Mhaske
- Lakshmi Soujanya Chandra 
- Jayani Rachapudi 

**Supervisor:** Dr. Tai Le Quy  
 
---

## Abstract

AI-powered recruitment systems are becoming increasingly common, offering efficiency in hiring processes. However, these systems may unintentionally perpetuate biases, leading to unfair hiring practices.
This study examines gender, racial, and age biases in recruitment systems using Large Language Models (LLMs) like BERT, GPT-2, and GPT-Neo. The research analyzes hiring decision datasets using machine
learning classification techniques and the Word Embedding Association Test (WEAT) to assess bias in AI-driven recruitment models. The findings reveal significant biases in gender and racial associations,
highlighting potential disparities in AI-driven hiring recommendations. This research underscores the need for bias mitigation strategies in AI-based recruitment tools, advocating for transparent and equitable hiring
practices.

---

## Methodology

### Dataset Overview
The dataset consists of 10,000 job applicants sourced from Kaggle. It includes:
- **Demographic attributes:** Age, gender, race, ethnicity.
- **Job-specific attributes:** Job roles applied for.
- **Textual data:** Resumes and job descriptions.
- **Outcome variable:** `Best Match` indicating whether a candidate's resume aligns with the job description.
You can find the dataset under the folder data, it also includes the alternate access to the dataset that is taken from kaggle.

### Data Preprocessing

#### Supervised Learning Preprocessing:

1.  **Encoding Techniques:**
    *   One-Hot Encoding for gender.
    *   Label Encoding for race categories.
    *   Frequency Encoding for job roles and ethnicity.

2.  **Skewness Transformation:**
    *   Applied Yeo-Johnson transformation to normalize skewed features.

3.  **Feature Scaling:**
    *   Used RobustScaler to standardize numerical features while minimizing outlier effects.

Further prepocessing of the dataset was done to ensure it can be used for the WEAT analysis.

#### WEAT Analysis Preprocessing:

1.  **Text Data Preprocessing**:
    *   Lowercasing to ensure uniformity.
    *   Removal of non-alphabetic characters.
    *   Tokenization and lemmatization using NLTK.
    *   Stopword removal to focus on meaningful terms.

2.  **Word set creation**:
    *   Defined STEM and non-STEM reference word lists to help for creation of word sets from dataset.
    *   Computed embeddings for these reference word lists and the data extracted our columns using TF-IDf. Followed by using cosine similarity to separate the data into STEM or Non STEM in terms of similiarity and reference list.
    *   The process is same but is done for the 3 models separately (BERT, GPT 2, GPT NEO) resulting in separate word sets representing how different models understand the language.


### Machine Learning Models
Two supervised learning models were employed:
1. **Logistic Regression**: For interpretability and linear relationships.
2. **Random Forest Classifier**: For capturing non-linear patterns and feature importance.

### Bias Analysis Tools
Bias was analyzed using the Word Embedding Association Test (WEAT).
Consider
– Target Sets (X,Y ): The concepts you’re testing for association. Example: X =e.g., programmer, engineer, scientist, ... Y = nurse, teacher, librarian, ...,
– Attribute Sets (A, B): The groups you’re comparing. Example: A = Male-associated words, B = Female-associated words.
- s(w, A, B) is the association score of a word w with the attribute sets A and B. We take the average of these scores for all words in X and subtract the average for all words in Y.
- The standard deviation measures how much the association scores vary across all words in X and Y. This ensures that the effect size is standardized, so it’s not influenced by the scale of the data.   

 <div style="text-align: center;">
  <img width="350" alt="image" src="https://github.com/user-attachments/assets/a3c7d107-accf-44cf-bfe1-542c6240e05f" />
 </div>
                 
- d > 0 → More association between X and A.
- d < 0 → More association between X and B.
- The range of this lies from -2 to 2
           
Below are the scenarios we considered for our project:
  
- Gender Bias Detection: This examines potential gender biases by analyzing the association between gendered names and STEM/non-STEM keywords.
  The male and female names are used as the attribute word sets.
  The WEAT effect size measures the association between the target word sets (STEM/non-STEM) and the attribute word sets (male/female names).
  
- First Name Bias (Race Associations): This investigates potential racial biases associated with first names.
  Unique first names are extracted from the 'Job Applicant Name' column. Names are categorized into White, Black, and Asian groups based on the 'Race' column.
  WEAT effect size is used to measure the association between all first names as the target word set and the racial groups (White, Black, Asian) as the attribute word sets.
  This analysis aims to determine if racial groups are differentially associated with first names in the embedding space.
  
- Race Bias: STEM/Non-STEM Associations: This detection is done to directly test racial bias in the association between race and STEM/non-STEM fields.
  Racial group names are used as attribute word sets, and stem_keywords and non_stem_keywords are used as target word sets, with WEAT effect sizes quantifying racial associations.
  
- Best Match Bias (STEM vs. Non-STEM Within Groups): Analyzes biases in "best match" classifications.
  Racial group names are used as attribute word sets, and names from STEM and non-STEM "best match" classifications are used as target word sets, measuring racial bias within these classifications.
  
- Age Bias: STEM/Non-STEM Associations: Examines age biases. Young and old names are used as attribute word sets, and stem_keywords and non_stem_keywords are used as target word sets, with WEAT effect sizes quantifying age associations.

---

## Findings

- Gender: All models show a reverse gender bias, associating STEM with female names, contradicting common stereotypes.
- Race: BERT exhibits strong pro-White biases, particularly in STEM fields and name associations. GPT NEO and GPT-2 show weaker, often reversed, biases favoring Black and Asian associations.
- Age: Minimal age bias is observed across all models.
- Model Comparison: BERT displays the strongest overall biases, likely due to its training data. GPT NEO and GPT-2 show weaker biases, possibly due to more diverse training or differing processing methods."

---

## Conclusion

The research highlights the ethical concerns surrounding AI-driven recruitment systems and emphasizes the importance of incorporating bias mitigation strategies. Recommendations include:
- Debiasing word embeddings.
- Fairness-aware model training methodologies.
- Real-time bias monitoring during hiring processes.

---

## Repository Structure

├── data/ # Directory for datasets
│ ├── job_applicant_dataset.csv # Your dataset file
| ├── alternate dataset access
├── Weat analysis/ # Jupyter notebooks for analysis
│ ├── BERT.ipynb # Notebook for BERT analysis
│ ├── GPT2.ipynb # Notebook for GPT-2 analysis
│ ├── GPTNeo.ipynb # Notebook for GPT-Neo analysis
├── results/ # results and visualizations
│ ├── AllPlots/ # Notebook containing the visualizations of the results
├── Supervised learning/ # Notebook for the machine learning aspect
│ ├── Supervised_learning.ipynb # Supervised learning with machine learning models
├── requirements.txt # necessary libraries 
├── README.md

---

## How to Use

1.  Clone this repository:

    ```
    git clone https://github.com/yourusername/recruitment-bias-analysis.git
    cd recruitment-bias-analysis
    ```
2.  Install dependencies:

    ```
    pip install -r requirements.txt
    ```
3.  Run Jupyter notebooks for preprocessing, modeling, or bias analysis:

    ```
    jupyter notebook notebooks/
    ```
4.  Explore results in the `results/` directory.

---

## Dependencies

- Core Libraries: numpy>=1.21.0 , pandas>=1.3.0 , scipy>=1.7.0 (for cosine distance)

- NLP Preprocessing: nltk>=3.6.3 # For stopwords, lemmatization

- Feature Extraction: scikit-learn>=0.24.2 # For TF-IDF

- Transformers: transformers>=4.31.0, tokenizers>=0.13.0 , torch>=1.10.0 # Backend for HuggingFace models

- Visualization: matplotlib>=3.4.2 , seaborn>=0.11.1

All the execution was done on the Jupyter Environment.


