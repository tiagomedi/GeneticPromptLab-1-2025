"""
GeneticPromptLab - Sistema de optimización genética para prompts
Sistema SSH-only usando Ollama (sin OpenAI)
"""

from .base_class import GeneticPromptLab
from .qa_optim import QuestionsAnswersOptimizer
from .utils import send_query_to_ollama

__version__ = "1.1.0"
__author__ = "GeneticPromptLab Team"
__description__ = "Sistema de optimización genética para prompts usando SSH/Ollama"

__all__ = ['GeneticPromptLab', 'QuestionsAnswersOptimizer', 'send_query_to_ollama']