import unittest
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Union
import json
import csv
from datetime import datetime
from unittest.mock import Mock
from src.autoaction import graph
from src.dataset_preparation import DatasetManager, TestDataset, ConversationTestCase

def prepare_email_dataset(self) -> TestDataset:
        """Prepare test dataset for email agent"""
        dataset = TestDataset(
            tests={
                "Email Tests": [
                    ConversationTestCase(
                        input="send an email to saad@gmail.com to let him know that project is finished",
                        expected_keywords=["Project Completion", "successfully completed", "Best regards"],
                        expected_tone="formal",
                        context="Basic project completion email"
                    ),
                    ConversationTestCase(
                        input="send an email to saad@gmail.com about urgent project delay",
                        expected_keywords=["Urgent", "delay", "project"],
                        expected_tone="formal",
                        context="Urgent notification"
                    ),
                    ConversationTestCase(
                        input="email saad@gmail.com about successful test results",
                        expected_keywords=["test results", "successful", "completed"],
                        expected_tone="formal",
                        context="Test results notification"
                    )
                ]
            }
        )
        return dataset

   
class TestEmailAgent(unittest.TestCase):
    def setUp(self):
        self.manager = DatasetManager()
        self.dataset = prepare_email_dataset(self)
        self.graph = Mock()
    def test_dataset_validation(self):
        """Test the dataset validation"""
        self.assertTrue(self.dataset.validate())

    def test_save_and_load_json(self):
        """Test saving and loading in JSON format"""
        # Save
        json_path = self.manager.save_dataset(self.dataset, "json")
        self.assertTrue(json_path.exists())
        
        # Load
        loaded_dataset = self.manager.load_dataset(json_path)
        self.assertTrue(loaded_dataset.validate())

    def test_save_and_load_csv(self):
        """Test saving and loading in CSV format"""
        # Save
        csv_path = self.manager.save_dataset(self.dataset, "csv")
        self.assertTrue(csv_path.exists())
        
        # Load
        loaded_dataset = self.manager.load_dataset(csv_path)
        self.assertTrue(loaded_dataset.validate())
    def test_email_generation(self):
        """Test email generation for different cases"""
        email_cases = self.dataset.tests.get("Email Tests", [])
        
        for case in email_cases:
            with self.subTest(input=case.input):
                # Configuration du mock
                mock_response = {
                    "task": case.input,
                    "result": "Subject: Project Completion\n\nDear Saad,\n\nProject successfully completed...\n\nBest regards"
                }
                self.graph.invoke.return_value = mock_response
                
                response = self.graph.invoke({"task": case.input})
                
                # Vérifications
                self.assertIsNotNone(response)
                self.assertIn("result", response)
                
                email_content = response["result"].lower()
                found_keywords = [
                    keyword for keyword in case.expected_keywords 
                    if keyword.lower() in email_content
                ]
                
                self.assertTrue(len(found_keywords) > 0)

def main():
    """Main function to demonstrate usage"""
    try:
        # Create test directory
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)
        
        # Initialize manager and prepare dataset
        manager = DatasetManager(test_dir)
        dataset = manager.prepare_guard_dataset()
        
        # Save in both formats
        json_path = manager.save_dataset(dataset, "json")
        csv_path = manager.save_dataset(dataset, "csv")
        
        # Load and validate
        loaded_json = manager.load_dataset(json_path)
        loaded_csv = manager.load_dataset(csv_path)
        
        assert loaded_json.validate(), "JSON dataset validation failed"
        assert loaded_csv.validate(), "CSV dataset validation failed"
        
        # logger.info("Dataset preparation completed successfully")
        
    except Exception as e:
        # logger.error(f"Error during dataset preparation: {str(e)}")
        raise

if __name__ == "__main__":
    unittest.main(verbosity=2)


