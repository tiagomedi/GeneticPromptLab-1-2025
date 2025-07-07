"""
GeneticPromptLab - Sistema de optimización genética para prompts
Incluye soporte para detección COVID usando Ollama vía SSH
"""

from .base_class import GeneticPromptLab
from .qa_optim import QuestionsAnswersOptimizer
from .utils import send_query2gpt

__version__ = "1.1.0"
__author__ = "GeneticPromptLab Team"
__description__ = "Sistema de optimización genética para prompts con detección COVID"

__all__ = ['GeneticPromptLab', 'QuestionsAnswersOptimizer', 'send_query2gpt']