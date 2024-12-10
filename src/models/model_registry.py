# src/models/model_registry.py

import os
import json
from datetime import datetime
from typing import Dict, Optional, List
import shutil

class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = registry_path
        self.metadata_file = os.path.join(registry_path, "model_metadata.json")
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize the model registry directory and metadata file."""
        os.makedirs(self.registry_path, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            self._save_metadata({})

    def _save_metadata(self, metadata: Dict):
        """Save metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def register_model(self, 
                      model_path: str, 
                      model_name: str,
                      version: str,
                      metrics: Dict[str, float],
                      description: str = "") -> str:
        """Register a new model version."""
        model_id = f"{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_dir = os.path.join(self.registry_path, model_id)
        os.makedirs(model_dir, exist_ok=True)

        target_path = os.path.join(model_dir, "model")
        if os.path.isdir(model_path):
            shutil.copytree(model_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, target_path)

        metadata = self._load_metadata()
        metadata[model_id] = {
            "name": model_name,
            "version": version,
            "metrics": metrics,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "path": target_path
        }
        self._save_metadata(metadata)
        
        return model_id

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model metadata by ID."""
        metadata = self._load_metadata()
        return metadata.get(model_id)

    def get_latest_model(self, model_name: str) -> Optional[Dict]:
        """Get the latest version of a model."""
        metadata = self._load_metadata()
        models = [
            (k, v) for k, v in metadata.items()
            if v["name"] == model_name and v["status"] == "active"
        ]
        if not models:
            return None
        return max(models, key=lambda x: x[1]["created_at"])[1]

    def list_models(self, model_name: Optional[str] = None) -> List[Dict]:
        """List all registered models or filter by name."""
        metadata = self._load_metadata()
        if model_name:
            return [
                {"id": k, **v}
                for k, v in metadata.items()
                if v["name"] == model_name
            ]
        return [{"id": k, **v} for k, v in metadata.items()]

    def archive_model(self, model_id: str):
        """Archive a model (mark as inactive)."""
        metadata = self._load_metadata()
        if model_id in metadata:
            metadata[model_id]["status"] = "archived"
            self._save_metadata(metadata)

    def delete_model(self, model_id: str):
        """Delete a model from the registry."""
        metadata = self._load_metadata()
        if model_id in metadata:
            model_path = os.path.join(self.registry_path, model_id)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            del metadata[model_id]
            self._save_metadata(metadata)