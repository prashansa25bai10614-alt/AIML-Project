# Fake Product Review Detection

This project uses basic Natural Language Processing (NLP) and machine learning to detect fake product reviews. It is designed to help users identify whether an online product review is genuine or fake using text classification.

## Features
- Loads a dataset of product reviews
- Preprocesses and vectorizes review text
- Trains a simple Logistic Regression classifier
- Evaluates model performance
- Predicts if a new review is fake or genuine

## Requirements
- Python 3.x
- pandas
- scikit-learn

Install dependencies with:
```
pip install pandas scikit-learn
```

## Dataset Format
The dataset should be a CSV file (e.g., `reviews.csv`) with the following columns:
- `review`: The text of the product review
- `label`: 1 for fake reviews, 0 for genuine reviews

Example:
| review                        | label |
|-------------------------------|-------|
| This product is great!        | 0     |
| Buy this now, best ever!      | 1     |

## Usage
1. Place your dataset (e.g., `reviews.csv`) in the project directory.
2. Run the script:
```
python fake_product_review_detection_codes.py
```
3. The script will output model accuracy and allow you to test predictions on new reviews.

## Example Output
```
Accuracy: 0.85
Classification Report:
 ...
Review: This product is amazing! Best purchase ever!
Prediction: Fake
```

## Project Structure
- `fake_product_review_detection_codes.py` — Main script for training and prediction
- `reviews.csv` — Your dataset (not included)

## Future Improvements
- Use advanced NLP techniques (TF-IDF, word embeddings)
- Try different machine learning models
- Deploy as a web application

## License
This project is for educational purposes.
