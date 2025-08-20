"""
Temporal Link (TLINK) predictor using the relation model from tlink_kaiwen.
This module integrates the trained relation classifier to predict temporal links
between events and time expressions, which can then be used to assist DSPy in
generating more accurate timelines.
"""

import json
import logging
import os
import re
import sys
from typing import List, Dict, Optional, Tuple, NamedTuple
import spacy
import medspacy
import torch
from transformers import BertTokenizer

# Add tlink_kaiwen to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tlink_kaiwen', 'src'))

try:
    from relation_model import RelationConfig, RelationDataset, RelationClassifier, load_config
    from test_relation_model import RelationModelTester
except ImportError as e:
    logging.warning(f"Failed to import tlink_kaiwen modules: {e}")
    logging.warning("TLINK prediction will be disabled")
    RelationModelTester = None

logger = logging.getLogger(__name__)

# Load spacy models for entity extraction
try:
    nlp = medspacy.load(enable=["medspacy_sentence_segmenter"])
except:
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
        logger.warning("No spaCy model available for entity extraction")


class Entity(NamedTuple):
    """Represents an extracted entity"""
    text: str
    start: int
    end: int
    entity_type: str  # 'EVENT' or 'TIMEX3'
    sentence_start: int
    sentence_end: int


class PredictedTLink(NamedTuple):
    """Represents a predicted temporal link"""
    subject_entity: Entity
    object_entity: Entity
    relation: str
    confidence: float
    source_sentence: str


class TLinkPredictor:
    """
    Predicts temporal links between events and time expressions using 
    the trained relation classifier from tlink_kaiwen.
    """
    
    def __init__(self, model_dir: str = None, base_model: str = 'dmis-lab/biobert-base-cased-v1.1'):
        """
        Initialize the TLINK predictor.
        
        Args:
            model_dir: Directory containing the trained relation model
            base_model: Base BERT model name
        """
        self.model_dir = model_dir
        self.base_model = base_model
        self.tester = None
        self.enabled = False
        
        if RelationModelTester is None:
            logger.warning("RelationModelTester not available - TLINK prediction disabled")
            return
            
        if model_dir and os.path.exists(model_dir):
            try:
                self.tester = RelationModelTester(model_dir, base_model)
                self.enabled = True
                logger.info(f"TLINK predictor initialized with model from {model_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize TLINK predictor: {e}")
        else:
            logger.warning(f"Model directory {model_dir} not found - TLINK prediction disabled")
    
    def is_enabled(self) -> bool:
        """Check if TLINK prediction is enabled"""
        return self.enabled and self.tester is not None
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract EVENT and TIMEX3 entities from text using simple heuristics.
        This is a simplified version - in practice, you'd want a trained NER model.
        """
        if not nlp:
            logger.warning("No spaCy model available for entity extraction")
            return []
        
        entities = []
        doc = nlp(text)
        
        # Common chemotherapy drug patterns
        chemo_patterns = [
            r'\b(?:chemotherapy|chemo|cyclophosphamide|docetaxel|carboplatin|paclitaxel|doxorubicin|adriamycin|bevacizumab|avastin|herceptin|tamoxifen|taxol|taxotere|gemcitabine|gemzar|cisplatin|methotrexate|abraxane|cytoxan|doxil)\b',
            r'\b(?:a\.?c|t\.?c|tch|ac)\b',
            r'\b(?:treatment|therapy|infusion|cycle|regimen|protocol)\b'
        ]
        
        # Time expression patterns (simplified)
        time_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # dates
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # ISO dates
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            r'\b\d{4}\b',  # years
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(?:week|month|year|day)s?\b',
            r'\b(?:today|yesterday|tomorrow|now|currently|recently|soon)\b',
            r'\b(?:after|before|during|until|since|from|to)\b'
        ]
        
        # Extract entities from each sentence
        for sent in doc.sents:
            sent_start = sent.start_char
            sent_end = sent.end_char
            sent_text = sent.text.strip()
            
            if not sent_text:
                continue
            
            # Extract EVENT entities (chemotherapy-related)
            for pattern in chemo_patterns:
                for match in re.finditer(pattern, sent_text, re.IGNORECASE):
                    entity_start = sent_start + match.start()
                    entity_end = sent_start + match.end()
                    entities.append(Entity(
                        text=match.group(),
                        start=entity_start,
                        end=entity_end,
                        entity_type='EVENT',
                        sentence_start=sent_start,
                        sentence_end=sent_end
                    ))
            
            # Extract TIMEX3 entities (time expressions)
            for pattern in time_patterns:
                for match in re.finditer(pattern, sent_text, re.IGNORECASE):
                    entity_start = sent_start + match.start()
                    entity_end = sent_start + match.end()
                    entities.append(Entity(
                        text=match.group(),
                        start=entity_start,
                        end=entity_end,
                        entity_type='TIMEX3',
                        sentence_start=sent_start,
                        sentence_end=sent_end
                    ))
        
        # Remove overlapping entities (keep longer ones)
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping the longer ones"""
        if not entities:
            return entities
        
        # Sort by start position
        entities = sorted(entities, key=lambda e: (e.start, -len(e.text)))
        
        non_overlapping = []
        for entity in entities:
            # Check if it overlaps with any already selected entity
            overlaps = False
            for selected in non_overlapping:
                if (entity.start < selected.end and entity.end > selected.start):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
    
    def predict_tlinks_from_file(self, text: str, entities_file: str, max_distance: int = 500) -> List[PredictedTLink]:
        """
        Predict temporal links using entities from a CSV file instead of extracting them.
        
        Args:
            text: Input clinical text
            entities_file: Path to CSV file containing predicted entities
            max_distance: Maximum character distance between entities to consider
            
        Returns:
            List of predicted temporal links
        """
        if not self.is_enabled():
            logger.debug("TLINK prediction disabled - returning empty list")
            return []
        
        # Load entities from file
        entities = self.load_entities_from_file(entities_file, text)
        
        if len(entities) < 2:
            logger.debug(f"Found only {len(entities)} entities - need at least 2 for relation prediction")
            return []
        
        logger.debug(f"Loaded {len(entities)} entities from file")
        
        # Use the same prediction logic as the regular method
        return self._predict_tlinks_with_entities(text, entities, max_distance)
    
    def load_entities_from_file(self, entities_file: str, text: str) -> List[Entity]:
        """
        Load predicted entities from a CSV file.
        
        Args:
            entities_file: Path to CSV file containing entities
            text: The text that entities were extracted from
            
        Returns:
            List of Entity objects
        """
        import pandas as pd
        
        entities = []
        
        try:
            df = pd.read_csv(entities_file)
            
            # Assume CSV has columns like: text, start, end, entity_type
            # Adjust column names as needed based on actual file format
            for _, row in df.iterrows():
                # Map the entity type to our expected format
                entity_type = 'EVENT' if row.get('entity_type', '').upper() in ['EVENT', 'TREATMENT', 'MEDICATION'] else 'TIMEX3'
                
                entity = Entity(
                    text=str(row.get('text', '')),
                    start=int(row.get('start', 0)),
                    end=int(row.get('end', 0)),
                    entity_type=entity_type,
                    sentence_start=max(0, int(row.get('start', 0)) - 100),  # Approximate sentence boundaries
                    sentence_end=min(len(text), int(row.get('end', 0)) + 100)
                )
                entities.append(entity)
                
        except Exception as e:
            logger.error(f"Error loading entities from file {entities_file}: {e}")
            return []
        
        return entities
    
    def _predict_tlinks_with_entities(self, text: str, entities: List[Entity], max_distance: int = 500) -> List[PredictedTLink]:
        """
        Internal method to predict temporal links given a list of entities.
        
        Args:
            text: Input clinical text
            entities: List of entities to use for prediction
            max_distance: Maximum character distance between entities to consider
            
        Returns:
            List of predicted temporal links
        """
        # Generate entity pairs within same sentence or close proximity
        entity_pairs = []
        
        for i, event_entity in enumerate(entities):
            for j, time_entity in enumerate(entities):
                if i == j:
                    continue
                
                # Check if entities are within reasonable distance
                distance = abs(event_entity.start - time_entity.start)
                if distance > max_distance:
                    continue
                
                # Prefer EVENT -> TIMEX3 relations, but allow both directions
                if (event_entity.entity_type == 'EVENT' and time_entity.entity_type == 'TIMEX3') or \
                   (event_entity.entity_type == 'TIMEX3' and time_entity.entity_type == 'EVENT'):
                    
                    # Create relation data in the format expected by the model
                    sentence_start = min(event_entity.sentence_start, time_entity.sentence_start)
                    sentence_end = max(event_entity.sentence_end, time_entity.sentence_end)
                    source_sentence = text[sentence_start:sentence_end].strip()
                    
                    if not source_sentence:
                        continue
                    
                    relation_data = {
                        "subject_text": event_entity.text,
                        "subject_start": event_entity.start,
                        "subject_end": event_entity.end,
                        "subject_class": 1 if event_entity.entity_type == 'EVENT' else 2,
                        "object_text": time_entity.text,
                        "object_start": time_entity.start,
                        "object_end": time_entity.end,
                        "object_class": 2 if time_entity.entity_type == 'TIMEX3' else 1,
                        "source_sentence": source_sentence,
                        "source_sentence_start": sentence_start,
                        "source_sentence_end": sentence_end,
                        "relation": "no_relation"  # Will be predicted
                    }
                    
                    entity_pairs.append((event_entity, time_entity, relation_data))
        
        if not entity_pairs:
            logger.debug("No valid entity pairs found for relation prediction")
            return []
        
        logger.debug(f"Generated {len(entity_pairs)} entity pairs for relation prediction")
        
        # Predict relations using the trained model
        test_data = [pair[2] for pair in entity_pairs]
        
        try:
            results = self.tester.test_model(test_data, batch_size=16, detailed_output=True)
            
            if "detailed_predictions" not in results:
                logger.warning("No detailed predictions returned from model")
                return []
            
            predictions = results["detailed_predictions"]
            
        except Exception as e:
            logger.error(f"Error during relation prediction: {e}")
            return []
        
        # Convert predictions to TLink objects
        tlinks = []
        
        for i, (event_entity, time_entity, relation_data) in enumerate(entity_pairs):
            if i >= len(predictions):
                break
            
            pred = predictions[i]
            predicted_relation = pred["predicted_relation"]
            confidence = pred["confidence"]
            
            # Only include relations that are not "no_relation" and have reasonable confidence
            if predicted_relation != "no_relation" and confidence > 0.5:
                tlink = PredictedTLink(
                    subject_entity=event_entity,
                    object_entity=time_entity,
                    relation=predicted_relation.upper(),  # Convert to uppercase (BEGINS-ON, etc.)
                    confidence=confidence,
                    source_sentence=relation_data["source_sentence"]
                )
                tlinks.append(tlink)
        
        logger.info(f"Predicted {len(tlinks)} temporal links from {len(entity_pairs)} entity pairs")
        
        return tlinks

    def predict_tlinks(self, text: str, max_distance: int = 500) -> List[PredictedTLink]:
        """
        Predict temporal links between entities in the given text.
        
        Args:
            text: Input clinical text
            max_distance: Maximum character distance between entities to consider
            
        Returns:
            List of predicted temporal links
        """
        if not self.is_enabled():
            logger.debug("TLINK prediction disabled - returning empty list")
            return []
        
        # Extract entities
        entities = self.extract_entities(text)
        
        if len(entities) < 2:
            logger.debug(f"Found only {len(entities)} entities - need at least 2 for relation prediction")
            return []
        
        logger.debug(f"Extracted {len(entities)} entities from text")
        
        # Use the internal method to predict with extracted entities
        return self._predict_tlinks_with_entities(text, entities, max_distance)
    
    def format_tlinks_for_dspy(self, tlinks: List[PredictedTLink]) -> str:
        """
        Format predicted temporal links as text that can be used as input to DSPy.
        
        Args:
            tlinks: List of predicted temporal links
            
        Returns:
            Formatted string describing the temporal relations
        """
        if not tlinks:
            return ""
        
        formatted_lines = []
        formatted_lines.append("PREDICTED TEMPORAL RELATIONS:")
        
        for tlink in sorted(tlinks, key=lambda x: x.confidence, reverse=True):
            confidence_str = f"(confidence: {tlink.confidence:.2f})"
            relation_str = f"'{tlink.subject_entity.text}' {tlink.relation.lower().replace('-', ' ')} '{tlink.object_entity.text}' {confidence_str}"
            formatted_lines.append(f"- {relation_str}")
        
        return "\n".join(formatted_lines)
    
    def extract_timeline_hints(self, tlinks: List[PredictedTLink]) -> Dict[str, List[str]]:
        """
        Extract timeline hints from predicted temporal links.
        
        Args:
            tlinks: List of predicted temporal links
            
        Returns:
            Dictionary mapping relation types to lists of (entity, time) pairs
        """
        hints = {
            "begins-on": [],
            "ends-on": [],
            "contains-1": []
        }
        
        for tlink in tlinks:
            relation_key = tlink.relation.lower().replace("-", "-")
            if relation_key in hints:
                hint = f"{tlink.subject_entity.text} -> {tlink.object_entity.text}"
                hints[relation_key].append(hint)
        
        return hints


def create_tlink_predictor(model_dir: str = None) -> TLinkPredictor:
    """
    Factory function to create a TLINK predictor.
    
    Args:
        model_dir: Directory containing the trained relation model
        
    Returns:
        TLinkPredictor instance
    """
    if model_dir is None:
        # Try to find a model directory in the tlink_kaiwen folder
        tlink_dir = os.path.join(os.path.dirname(__file__), '..', 'tlink_kaiwen')
        potential_dirs = ['models', 'trained_model', 'relation_model']
        
        for potential_dir in potential_dirs:
            full_path = os.path.join(tlink_dir, potential_dir)
            if os.path.exists(full_path) and any(f.endswith('.pt') for f in os.listdir(full_path)):
                model_dir = full_path
                break
    
    return TLinkPredictor(model_dir)


if __name__ == "__main__":
    # Test the TLINK predictor
    logging.basicConfig(level=logging.INFO)
    
    # Sample clinical text
    test_text = """
    Patient started chemotherapy in August 2011. Cyclophosphamide was administered on 8/8/2011.
    Docetaxel treatment began on the same day. The patient completed chemotherapy on 10/10/2011.
    Follow-up appointments were scheduled for December 2011.
    """
    
    predictor = create_tlink_predictor()
    
    if predictor.is_enabled():
        print("Testing TLINK prediction...")
        tlinks = predictor.predict_tlinks(test_text)
        
        print(f"\nFound {len(tlinks)} temporal links:")
        for tlink in tlinks:
            print(f"- {tlink.subject_entity.text} {tlink.relation} {tlink.object_entity.text} (confidence: {tlink.confidence:.2f})")
        
        print("\nFormatted for DSPy:")
        formatted = predictor.format_tlinks_for_dspy(tlinks)
        print(formatted)
        
        print("\nTimeline hints:")
        hints = predictor.extract_timeline_hints(tlinks)
        for relation, hint_list in hints.items():
            if hint_list:
                print(f"{relation}: {hint_list}")
    else:
        print("TLINK prediction is disabled (no trained model found)")