# src/models/evaluate.py

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import logging
from typing import Dict, List, Tuple
import os
from datetime import datetime
from src.models.training import SentimentClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str):
        """Initialize evaluator with model path."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load model and tokenizer
        self.model = SentimentClassifier.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def predict_batch(self, texts: List[str], numerical_features: torch.Tensor) -> np.ndarray:
        """Make predictions for a batch of texts."""
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        numerical_features = numerical_features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, numerical_features=numerical_features)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        return predictions.cpu().numpy()
    
    def evaluate_model(self, test_data_path: str) -> Dict:
        """Evaluate model on test dataset."""
        logger.info("Starting model evaluation...")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # Prepare numerical features
        numerical_cols = [col for col in test_df.columns if col.startswith(('tfidf_', 'sentiment_', 'score_'))]
        numerical_features = torch.FloatTensor(test_df[numerical_cols].values)
        
        # Get predictions
        predictions = []
        batch_size = 32
        
        for i in range(0, len(test_df), batch_size):
            batch_texts = test_df['processed_text'][i:i + batch_size].tolist()
            batch_numerical = numerical_features[i:i + batch_size]
            batch_predictions = self.predict_batch(batch_texts, batch_numerical)
            predictions.extend(batch_predictions)
        
        predictions = np.array(predictions)
        
        # Convert string labels to numeric
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        true_labels = test_df['sentiment'].map(label_map).values
        
        # Calculate metrics
        report = classification_report(
            true_labels,
            predictions,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Save visualizations
        self._save_confusion_matrix(cm)
        self._analyze_errors(test_df, predictions, true_labels)
        
        return report
    
    def _save_confusion_matrix(self, cm: np.ndarray):
        """Create and save confusion matrix visualization."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save plot
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig('reports/figures/confusion_matrix.png')
        plt.close()
    
    def _analyze_errors(self, test_df: pd.DataFrame, predictions: np.ndarray, true_labels: np.ndarray):
        """Analyze and save error examples."""
        # Find misclassified examples
        mask = predictions != true_labels
        misclassified_df = test_df[mask].copy()
        misclassified_df['predicted_label'] = [self.id2label[p] for p in predictions[mask]]
        
        # Group errors by true and predicted labels
        error_analysis = []
        for true_label in ['negative', 'neutral', 'positive']:
            for pred_label in ['negative', 'neutral', 'positive']:
                if true_label != pred_label:
                    examples = misclassified_df[
                        (misclassified_df['sentiment'] == true_label) & 
                        (misclassified_df['predicted_label'] == pred_label)
                    ]
                    if len(examples) > 0:
                        error_analysis.append({
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'count': len(examples),
                            'examples': examples['text'].head(3).tolist()
                        })
        
        # Save error analysis
        os.makedirs('reports', exist_ok=True)
        with open('reports/error_analysis.txt', 'w') as f:
            f.write("Error Analysis Report\n")
            f.write("===================\n\n")
            for error in error_analysis:
                f.write(f"\nTrue: {error['true_label']}, Predicted: {error['predicted_label']}\n")
                f.write(f"Count: {error['count']}\n")
                f.write("Example texts:\n")
                for i, example in enumerate(error['examples'], 1):
                    f.write(f"{i}. {example}\n")
                f.write("-" * 80 + "\n")

def main():
    try:
        # Get latest model
        model_dirs = [d for d in os.listdir('models') if d.startswith('sentiment_model_')]
        latest_model = max(model_dirs)
        model_path = os.path.join('models', latest_model, 'final_model')
        
        # Get latest test data
        data_files = [f for f in os.listdir('data/processed') if f.startswith('engineered_features_')]
        latest_data = max(data_files)
        data_path = os.path.join('data/processed', latest_data)
        
        logger.info(f"Using model: {model_path}")
        logger.info(f"Using data: {data_path}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path)
        
        # Evaluate model
        metrics = evaluator.evaluate_model(data_path)
        
        # Print results
        print("\nModel Evaluation Results:")
        print("========================")
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print("\nPer-class metrics:")
        for label in ['negative', 'neutral', 'positive']:
            print(f"\n{label.title()}:")
            print(f"Precision: {metrics[label]['precision']:.4f}")
            print(f"Recall: {metrics[label]['recall']:.4f}")
            print(f"F1-score: {metrics[label]['f1-score']:.4f}")
        
        print("\nDetailed reports have been saved to 'reports' directory")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()