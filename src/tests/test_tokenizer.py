import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.tokenizer import BillTokenizer

@pytest.fixture
def tokenizer():
    return BillTokenizer()

def test_empty_text(tokenizer):
    """Test 1: empty strings should not crash and return empty lists."""
    tokens, tags = tokenizer.align_bio("", [])
    assert tokens == []
    assert tags == []

def test_unicode_and_emojis(tokenizer):
    """Test 2: Emojis and weird unicode shouldn't break alignment."""
    text = "Ayam Bakar 🍗 Rp15.000"
    entities = [
        {"start": 0, "end": 10, "label": "ITEM"},
        {"start": 13, "end": 21, "label": "PRICE"}
    ]
    tokens, tags = tokenizer.align_bio(text, entities)
    
    assert "Ayam" in tokens
    assert "Bakar" in tokens
    assert "Rp15.000" in tokens
    assert len(tokens) == len(tags)

def test_overlapping_entities(tokenizer):
    """Test 3: If annotations are overlapping/messy, it shouldn't crash."""
    text = "Nasi Goreng"
    entities = [
        {"start": 0, "end": 11, "label": "ITEM"},
        {"start": 5, "end": 11, "label": "WRONG_ITEM"}
    ]
    tokens, tags = tokenizer.align_bio(text, entities)
    
    assert tokens == ["Nasi", "Goreng"]
    assert len(tokens) == len(tags)
    assert tags == ["B-ITEM", "B-WRONG_ITEM"]