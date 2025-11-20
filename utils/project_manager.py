import os
import json
from pathlib import Path

class ProjectManager:
    """Manages video generation projects and metadata."""
    def __init__(self, projects_dir='projects'):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(exist_ok=True)
    
    def create_project(self, name: str, prompt: str, config: dict) -> str:
        """Create new project directory with metadata."""
        project_path = self.projects_dir / name
        project_path.mkdir(exist_ok=True)
        
        metadata = {
            'name': name,
            'prompt': prompt,
            'config': config,
            'status': 'initialized'
        }
        
        metadata_path = project_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create subdirectories
        (project_path / 'output').mkdir(exist_ok=True)
        (project_path / 'scripts').mkdir(exist_ok=True)
        (project_path / 'assets').mkdir(exist_ok=True)
        
        return str(project_path)
    
    def update_status(self, project_name: str, status: str):
        """Update project status."""
        metadata_path = self.projects_dir / project_name / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['status'] = status
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def list_projects(self) -> list:
        """List all projects."""
        return [d.name for d in self.projects_dir.iterdir() if d.is_dir()]
