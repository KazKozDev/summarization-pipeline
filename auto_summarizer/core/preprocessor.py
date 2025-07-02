"""
Document preprocessing module for text summarization.
Handles text cleaning, tokenization, and other NLP preprocessing tasks.
"""
import re
import string
from typing import List, Tuple, Dict, Any
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class DocumentPreprocessor:
    """
    Handles document preprocessing tasks including text cleaning,
    tokenization, and other NLP preprocessing steps.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize the document preprocessor.
        
        Args:
            language: Language of the text (default: "english")
        """
        self.language = language
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)
        
        # Download required NLTK data
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        return sent_tokenize(text, language=self.language)
    
    def preprocess_sentence(self, sentence: str) -> List[str]:
        """
        Preprocess a single sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of processed tokens
        """
        if not sentence:
            return []
            
        # Tokenize
        tokens = word_tokenize(sentence)
        
        # Remove stopwords and punctuation, lemmatize
        processed_tokens = []
        for word, tag in pos_tag(tokens):
            # Convert to lowercase and remove punctuation
            word = word.lower()
            word = word.strip()
            
            # Skip if empty, stopword, or punctuation
            if (not word or 
                word in self.stop_words or 
                all(char in self.punctuation for char in word)):
                continue
                
            # Get part of speech for lemmatization
            wordnet_pos = self._get_wordnet_pos(tag) if tag else 'n'
            
            # Lemmatize
            word = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
            processed_tokens.append(word)
            
        return processed_tokens
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Map treebank POS tags to WordNet POS tags.
        
        Args:
            treebank_tag: Treebank POS tag
            
        Returns:
            WordNet POS tag
        """
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
    
    def get_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity, entity_type) tuples
        """
        if not text:
            return []
            
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """
        Process a complete document and return structured information.
        
        Args:
            text: Input document text
            
        Returns:
            Dictionary containing processed document information
        """
        if not text:
            return {
                'original_text': '',
                'cleaned_text': '',
                'sentences': [],
                'tokens': [],
                'processed_sentences': [],
                'named_entities': []
            }
            
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = self.tokenize_sentences(cleaned_text)
        
        # Process each sentence
        processed_sentences = [self.preprocess_sentence(sent) for sent in sentences]
        
        # Flatten tokens
        tokens = [token for sent in processed_sentences for token in sent]
        
        # Get named entities
        named_entities = self.get_named_entities(text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'tokens': tokens,
            'processed_sentences': processed_sentences,
            'named_entities': named_entities
        }
