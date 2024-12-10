# src/models/ab_testing.py

import random
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelVariant:
    model_id: str
    weight: float = 1.0
    
class ABTestingManager:
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
        
    def create_test(self, 
                    test_id: str,
                    variants: List[ModelVariant],
                    description: str = ""):
        """Create a new A/B test."""
        total_weight = sum(v.weight for v in variants)
        normalized_variants = [
            ModelVariant(v.model_id, v.weight/total_weight)
            for v in variants
        ]
        
        self.active_tests[test_id] = {
            "variants": normalized_variants,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "results": {v.model_id: [] for v in variants}
        }
        
    def select_model(self, test_id: str) -> Optional[str]:
        """Select a model variant based on weights."""
        if test_id not in self.active_tests:
            return None
            
        variants = self.active_tests[test_id]["variants"]
        weights = [v.weight for v in variants]
        return np.random.choice([v.model_id for v in variants], p=weights)
    
    def record_result(self,
                     test_id: str,
                     model_id: str,
                     metrics: Dict[str, float]):
        """Record the result of a prediction."""
        if test_id in self.active_tests:
            self.active_tests[test_id]["results"][model_id].append({
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            })
            
    def get_test_results(self, test_id: str) -> Dict:
        """Get statistical results for an A/B test."""
        if test_id not in self.active_tests:
            return {}
            
        results = {}
        test_data = self.active_tests[test_id]
        
        for model_id, model_results in test_data["results"].items():
            if not model_results:
                continue
                
            metrics = {}
            for metric in model_results[0]["metrics"].keys():
                values = [r["metrics"][metric] for r in model_results]
                metrics[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values),
                    "median": np.median(values),
                    "p95": np.percentile(values, 95)
                }
            
            results[model_id] = metrics
            
        return results
    
    def end_test(self, test_id: str):
        """End an A/B test and store its results."""
        if test_id in self.active_tests:
            self.test_results[test_id] = {
                **self.active_tests[test_id],
                "ended_at": datetime.now().isoformat(),
                "final_results": self.get_test_results(test_id)
            }
            del self.active_tests[test_id]