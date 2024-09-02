# Spam Email Detector

This project aims to build a machine learning model that can detect whether an email is spam or not. The model is trained on a diverse dataset of email messages and their corresponding labels (spam or valid).

## Code Summary:

1. **Import Libraries:**
   - **pandas**: Used for data manipulation and analysis.
   - **sklearn**: A machine learning library that provides tools for model training, evaluation, and preprocessing.
   - **nltk**: The Natural Language Toolkit, used for text processing.
   - **re**: A library for regular expression operations, useful for text cleaning.

2. **Download NLTK Resources:**
   - Downloads the stopwords and WordNet lemmatizer data from NLTK, which are essential for text preprocessing.

3. **Load Dataset:**
   - Reads the CSV file `spam_ham_dataset.csv` into a pandas DataFrame named `data`. This dataset should contain two columns: `email` (the email messages) and `label` (indicating whether each email is spam or valid).

4. **Define Preprocessing Function:**
   - `preprocess_text`: A function that takes a text string as input and performs several preprocessing steps:
     - **Remove Special Characters and Digits**: Uses regular expressions to remove non-word characters and digits.
     - **Convert to Lowercase**: Converts all characters to lowercase to ensure uniformity.
     - **Remove Stopwords**: Filters out common words (like "and", "the", etc.) that do not contribute significant meaning.
     - **Lemmatization**: Reduces words to their base or root form (e.g., "running" becomes "run").

5. **Preprocess Dataset:**
   - Applies the `preprocess_text` function to the `email` column of the DataFrame, updating it with the cleaned text.

6. **Define Features and Labels:**
   - `X`: Contains the preprocessed email messages (features).
   - `y`: Contains the corresponding labels (spam or valid).

7. **Split Dataset:**
   - Uses `train_test_split` to divide the dataset into training and testing sets.
   - `test_size=0.3` indicates that 30% of the data will be used for testing.
   - `stratify=y` ensures that the split maintains the same proportion of spam and valid emails in both training and testing sets.

8. **Create a Pipeline:**
   - A pipeline is created to streamline the process of transforming the data and training the model.
   - The pipeline consists of:
     - **TfidfVectorizer**: Converts the text data into TF-IDF features.
     - **MultinomialNB**: A Naive Bayes classifier suitable for text classification.

9. **Determine Cross-Validation Splits:**
   - Calculates the minimum class size in the training set to determine the number of splits for cross-validation.
   - Ensures that the number of splits does not exceed the smallest class size.

10. **Cross-Validation:**
    - Uses `StratifiedKFold` to create balanced splits for cross-validation.
    - `cross_val_score` evaluates the model's accuracy on the training set using the defined pipeline.

11. **Hyperparameter Tuning:**
    - Defines a parameter grid for tuning:
      - **ngram_range**: Tests both unigrams and bigrams.
      - **alpha**: Tests different smoothing parameters for the Naive Bayes classifier.
    - `GridSearchCV` is used to find the best combination of parameters based on cross-validated accuracy.

12. **Retrieve Best Model:**
    - After the grid search, the best model (with the optimal hyperparameters) is stored in `best_model`.

13. **Model Evaluation:**
    - Uses the best model to make predictions on the test set.

14. **Display Results:**
    - Calculates and prints various evaluation metrics:
      - **Accuracy**: Overall accuracy of the model on the test set.
      - **Classification Report**: Detailed metrics including precision, recall, and F1-score for each class.
      - **Confusion Matrix**: A matrix showing the true positives, false positives, true negatives, and false negatives.

15. **Prediction Function:**
    - Defines a function `predict_spam` that takes an email message as input:
      - Preprocesses the email message using the `preprocess_text` function.
      - Uses the trained model to predict whether the email is spam or valid.
      - Returns the predicted class.

16. **Main Execution Block:**
    - If the script is run directly, it provides an example email message to classify. It calls the `predict_spam` function and prints the classification result.

This code implements a spam email detection system using machine learning techniques. It includes data preprocessing, model training, evaluation, and a prediction function, all structured in a clear and logical manner. The use of pipelines, cross-validation, and hyperparameter tuning helps ensure that the model is robust and performs well on unseen data.

## Features
- **Data Preprocessing**: The code includes functions to preprocess the email messages, such as removing special characters, digits, converting to lowercase, removing stopwords, and performing lemmatization.
- **Model Training**: The project uses a Naive Bayes classifier with TF-IDF vectorization to train the spam detection model. It employs cross-validation and hyperparameter tuning to optimize the model's performance.
- **Prediction Function**: The code includes a function `predict_spam` that takes an email message as input and returns whether it is classified as spam or not.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- nltk (Natural Language Toolkit)

## Usage
1. **Prepare the Dataset**: Ensure that you have a CSV file named `spam_ham_dataset.csv` containing the email messages and their corresponding labels. The file should have two columns: `email` and `label`.
2. **Run the Code**: Execute the provided Python script in your environment. The code will:
   - Load the dataset
   - Preprocess the email messages
   - Split the data into training and testing sets
   - Train the spam detection model using Naive Bayes and TF-IDF
   - Perform cross-validation and hyperparameter tuning
   - Evaluate the model's performance on the test set
   - Print the results, including cross-validation accuracy, best parameters, test set accuracy, classification report, and confusion matrix
3. **Use the Prediction Function**: After running the code, you can use the `predict_spam` function to classify new email messages as spam or not. The function takes an email message as input and returns the predicted label.

## Enhancements
- **Diverse Dataset**: The code includes a sample diverse dataset with a variety of spam and valid email messages. You can replace this with your own dataset or expand the existing one to improve the model's performance.
- **Hyperparameter Tuning**: The code uses `GridSearchCV` to tune the hyperparameters of the TF-IDF vectorizer and Naive Bayes classifier. You can experiment with additional hyperparameters or try different combinations to further optimize the model.
- **Model Evaluation**: The code evaluates the model's performance using accuracy, classification report, and confusion matrix. You can explore other evaluation metrics or customize the evaluation process based on your specific requirements.

## Future Work
- **Implement Additional Preprocessing Techniques**: Explore other preprocessing techniques, such as stemming, named entity recognition, or sentiment analysis, to enhance the model's understanding of email content.
- **Try Different Machine Learning Algorithms**: Experiment with other classification algorithms, such as Support Vector Machines (SVM), Random Forest, or Deep Learning models, to compare their performance with the Naive Bayes classifier.
- **Incorporate Email Metadata**: Consider incorporating additional features from the email metadata, such as sender information, subject line, or email headers, to improve the model's accuracy.
- **Deploy the Model**: Integrate the spam detection model into a real-world application or email service to provide spam filtering capabilities.

## Conclusion
This project demonstrates a basic implementation of a spam email detector using machine learning techniques. By leveraging a diverse dataset, preprocessing, and model optimization, the code provides a foundation for building an effective spam detection system.
