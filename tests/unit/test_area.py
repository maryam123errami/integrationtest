from langchain.schema import HumanMessage
import pytest
from src.area import graph
import sys
sys.path.append('./src')
# Cas de test prédéfinis
test_cases = [
    {"input": "definition of machine learning", "expected": "Researcher"},
    {"input": "scrape this page: https://medium.com/data-science-at-microsoft/how-large-language-models-work-91c362f5b78f", "expected": "WebScraper"},
]

def test_run():
    decisions = []
    expected_decisions = []
    
    for case in test_cases:
        # Exécute le test
        result = graph.invoke({"messages": [HumanMessage(content=case["input"])]})
        name_value = result['messages'][1].name
        
        # Affiche les détails du test
        print(f"Test input: {case['input']}")
        print(f"Expected: {case['expected']}, Got: {name_value}")
        
        # Vérifie si le résultat correspond à l'attendu
        try:
            assert name_value == case["expected"], f"Test failed: Expected {case['expected']}, but got {name_value}"
            print("✅ Test passed")
        except AssertionError as e:
            print(f"❌ {str(e)}")
        
        # Stocke les résultats pour le calcul de précision
        decisions.append(name_value)
        expected_decisions.append(case["expected"])
    
    # Calcul de la précision
    correct_count = sum(1 for d, e in zip(decisions, expected_decisions) if d == e)
    total_count = len(decisions)
    precision_rate = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    # Vérifie le taux de précision minimal attendu (par exemple 80%)
    try:
        assert precision_rate >= 80, f"Precision rate too low: {precision_rate:.2f}%"
        print(f"\n✅ Overall accuracy rate: {precision_rate:.2f}%")
    except AssertionError as e:
        print(f"\n❌ {str(e)}")
    
    return precision_rate

if __name__ == "__main__":
    test_run()