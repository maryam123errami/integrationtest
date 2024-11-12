from dataclasses import dataclass, field
from typing import List, Dict, Union
from pathlib import Path
import unittest
import json
import csv
from datetime import datetime
from agent import agent_executor  # Make sure this is correctly implemented
from langchain.schema import HumanMessage
from dataset_preparation import DatasetManager, TestDataset, ConversationTestCase

    

def prepare_KB_dataset(self) -> TestDataset:
        kb_dataset = TestDataset(
            tests={
                "Basic Operations": [
                    ConversationTestCase(
                        input="Add a new document about Python programming",
                        # expected_tool="add_kb",
                        expected_keywords=["added"],
                        context="Adding new knowledge base"
                    ),
                    ConversationTestCase(
                        input="Show me all my knowledge bases",
                        # expected_tool="list_kb",
                        expected_keywords=["knowledge bases"],
                        context="Listing knowledge bases"
                    ),
                    ConversationTestCase(
                        input="Find information about machine learning",
                        # expected_tool="find_kb",
                        expected_keywords=["add"],
                        context="Finding specific knowledge"
                    )
                ]
            }
        )
        return kb_dataset


class KBAgentTests(unittest.TestCase):
    def setUp(self):
        self.manager = DatasetManager(Path("test_data"))
        self.dataset = prepare_KB_dataset(self)
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
    def test_addKB(self):
        """Basic Operations"""
        input_cases = self.dataset.tests.get("Basic Operations", [])
        for case in input_cases:
            with self.subTest(input=case.input):
                result = agent_executor({
                    "messages": [HumanMessage(content=case.input)]
                })
                self.assertIsNotNone(result, "Result should not be None")
                if 'output' in result:
                    output = result['output'].lower()
                    for expected in case.expected_keywords:
                        self.assertIn(expected.lower(), output, f"Expected '{expected}' in output for input '{case.input}'")
                elif 'messages' in result:
                    messages = result['messages']
                    self.assertTrue(any(expected.lower() in msg.lower() for expected in case.expected_keywords for msg in messages), "Expected response not found in messages")

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



