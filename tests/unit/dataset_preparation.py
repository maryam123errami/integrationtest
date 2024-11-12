from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Union
import json
import csv
from datetime import datetime

@dataclass
class ConversationTestCase:
    input: str
    expected_keywords: List[str]
    expected_tone: str = "formal"
    context: str = ""

    def to_dict(self):
        return {
            "input": self.input,
            "expected_keywords": self.expected_keywords,
            "expected_tone": self.expected_tone,
            "context": self.context
        }

@dataclass
class TestDataset:
    tests: Dict[str, List[ConversationTestCase]]
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return {
            "tests": {
                category: [test.to_dict() for test in test_cases]
                for category, test_cases in self.tests.items()
            },
            "creation_date": self.creation_date
        }

    def validate(self) -> bool:
        """Validate the dataset structure and content"""
        if not isinstance(self.tests, dict):
            return False
        
        for category, test_cases in self.tests.items():
            if not isinstance(test_cases, list):
                return False
            for test_case in test_cases:
                if not isinstance(test_case, ConversationTestCase):
                    return False
        return True

class DatasetManager:
    def __init__(self, base_path: Path = Path(".")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_dataset(self, dataset: TestDataset, format: str = "json") -> Path:
        """Save the dataset in the specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            file_path = self.base_path / f"test_dataset_{timestamp}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)
        
        elif format.lower() == "csv":
            file_path = self.base_path / f"test_dataset_{timestamp}.csv"
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["category", "input", "expected_keywords", "expected_tone", "context"])
                
                for category, test_cases in dataset.tests.items():
                    for test_case in test_cases:
                        writer.writerow([category, test_case.input, ",".join(test_case.expected_keywords), test_case.expected_tone, test_case.context])
        else:
            raise ValueError(f"Format not supported: {format}")
            
        return file_path

    def load_dataset(self, file_path: Union[str, Path]) -> TestDataset:
        """Load the dataset from a file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File dataset not found: {file_path}")
            
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tests = {
                    category: [
                        ConversationTestCase(**test_case)
                        for test_case in test_cases
                    ]
                    for category, test_cases in data['tests'].items()
                }
                return TestDataset(tests=tests, creation_date=data.get('creation_date'))
                
        elif suffix == '.csv':
            tests = {}
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    category = row['category']
                    if category not in tests:
                        tests[category] = []
                    
                    test_case = ConversationTestCase(
                        input=row['input'],
                        expected_keywords=row['expected_keywords'].split(',') if row['expected_keywords'] else [],
                        expected_tone=row['expected_tone'],
                        context=row['context']
                    )
                    tests[category].append(test_case)
                    
            return TestDataset(tests=tests)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
def prepare_email_dataset() -> TestDataset:
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
