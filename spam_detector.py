import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv('spam_dataset.csv')  

# Data preprocessing function
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Apply preprocessing to the dataset
data['email'] = data['email'].apply(preprocess_text)

# Split the dataset into features and labels
X = data['email']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build a pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  
    ('classifier', MultinomialNB())
])

# Determine the number of splits for cross-validation based on class distribution
min_class_size = y_train.value_counts().min()
n_splits = min(5, min_class_size)  # Ensure n_splits does not exceed the smallest class size

# Cross-validation to evaluate the model
skf = StratifiedKFold(n_splits=n_splits)  # Use StratifiedKFold for balanced splits
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams vs. unigrams+bigrams
    'classifier__alpha': [0.1, 0.5, 1.0, 1.5]  # Smoothing parameter for Naive Bayes
}
grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy')  # Use the same StratifiedKFold for grid search
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)
confusion_matrix_output = confusion_matrix(y_test, y_pred)

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Set Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_output)
print("Confusion Matrix:\n", confusion_matrix_output)

# Function to predict if an email is spam or not
def predict_spam(email_message):
    # Preprocess the input email message
    processed_email = preprocess_text(email_message)
    
    # Make prediction
    prediction = best_model.predict([processed_email])
    
    return prediction[0]  # Return the predicted class

# Example usage of the prediction function
if __name__ == "__main__":
    test_email = "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim your prize."
    result = predict_spam(test_email)
    print(f"The email is classified as: {result}")