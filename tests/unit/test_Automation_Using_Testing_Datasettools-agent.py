from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import json
import csv
import unittest
from pathlib import Path
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage
from src.dataset_preparation import DatasetManager, TestDataset, ConversationTestCase
from src.tools_agents_build import graph

def prepare_dataset(self) -> TestDataset:
        """Prepare the complete test dataset"""
        dataset = TestDataset(
            tests={
                "Input Variation Testing": [
                    ConversationTestCase(
                        input="Hello, what are the available delivery options?",
                        expected_keywords=["delivery", "options"],
                        expected_tone="informal",
                        context="Question about delivery options"
                    ),
                    #  ConversationTestCase(
                    #     input="Could you please provide me with the opening hours?",
                    #     expected_keywords=["hours", "opening"],
                    #     expected_tone="formal",
                    #     context="Request for hours"
                    # ),
                    ConversationTestCase(
                        input="What types of delivery do you offer?",
                        expected_keywords=["types", "delivery"],
                        expected_tone="informal",
                        context="Inquiry about delivery types"
                    ),
                    # ConversationTestCase(
                    #     input="What are your opening hours?",
                    #     expected_keywords=["hours", "opening"],
                    #     expected_tone="formal",
                    #     context="Request for hours"
                    # ),
                    ConversationTestCase(
                        input="<p>What is your return policy?</p>",
                        expected_keywords=["policy", "return"],
                        expected_tone="informal",
                        context="Question about the return policy"
                    ),
                    ConversationTestCase(
                        input='{"request": "What are the delivery times?"}',
                        expected_keywords=["times", "delivery"],
                        expected_tone="informal",
                        context="Inquiry about delivery times"
                    ),
                    # ConversationTestCase(
                    #     input="OK",
                    #     expected_keywords=["Hello"],
                    #     expected_tone="simple",
                    #     context="Very short input"
                    # ),
                    ConversationTestCase(
                        input="I would like to understand in detail how your delivery service works and the associated times.",
                        expected_keywords=["understand", "details", "service", "delivery"],
                        expected_tone="formal",
                        context="Detailed request about the service"
                    ),
                    ConversationTestCase(
                        input="Can you explain what circular economy is?",
                        expected_keywords=["explain", "circular economy"],
                        expected_tone="informal",
                        context="Question about circular economy"
                    ),
                    ConversationTestCase(
                        input="What is circular economy?",
                        expected_keywords=["circular economy"],
                        expected_tone="informal",
                        context="Definition request"
                    ),
                ],
                
                "Edge case handling": [
                    ConversationTestCase(
                        input="",
                        expected_keywords=[],
                        expected_tone="simple",
                        context="Empty input"
                    ),
                    ConversationTestCase(
                        input="!@#$%^&*()_+{}:\"<>?",  # Input with special characters
                        expected_keywords=[],
                        expected_tone="simple",
                        context="Input with special characters"
                    ),
                    ConversationTestCase(
                        input=" " * 100,  # Input with spaces only
                        expected_keywords=[],
                        expected_tone="simple",
                        context="Input with spaces only"
                    ),
                    ConversationTestCase(
                        input="🌟💫✨",  # Emojis
                        expected_keywords=["unlock", "account"],
                        expected_tone="formal",
                        context="Request to unlock an account"
                    ),
                    ConversationTestCase(
                        input="我需要帮助我的订单",  # Input in unsupported language (Chinese)
                        expected_keywords=[],
                        expected_tone="simple",
                        context="Input in unsupported language"
                    ),
                ]
            }
        )
        return dataset

class TestDatasetTests(unittest.TestCase):
    def setUp(self):
        self.manager = DatasetManager(Path("test_data"))
        self.dataset = prepare_dataset(self)

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

    def test_edge_cases(self):
        """Test edge cases handling"""
        edge_cases = self.dataset.tests.get("Edge case handling", [])
        
        for case in edge_cases:
            with self.subTest(case=case):
                result = graph.invoke({
                    "messages": [HumanMessage(content=case.input)]
                })
                self.assertIsNotNone(result)
    def test_input_variations(self):
        """Test input variations"""
        
            
        input_cases = self.dataset.tests.get("Input Variation Testing", [])
        
        for case in input_cases:
            with self.subTest(input=case.input):
                result = graph.invoke({
                    "messages": [HumanMessage(content=case.input)]
                })
                self.assertIsNotNone(result, "Le résultat ne doit pas être None")
                
                response = result["messages"][-1]
                response_content = response.content.lower()
                
                found_keywords = [
                    keyword for keyword in case.expected_keywords 
                    if keyword.lower() in response_content
                ]
                
                self.assertTrue(
                    len(found_keywords) > 0,
                    f"responce  don't have any kywords : {case.input}\n"
                    f"keywords: {case.expected_keywords}\n"
                    f"Responce: {response_content[:100]}..."
                )
def main():
    """Main function to demonstrate usage"""
    try:
        # Create test directory
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)
        
        # Initialize manager and prepare dataset
        manager = DatasetManager(test_dir)
        dataset = manager.prepare_dataset()
        
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

