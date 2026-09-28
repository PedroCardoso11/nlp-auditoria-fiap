"""Módulo de Pré-processamento e Limpeza de Texto para NLP.

Contém funções reutilizáveis para higienização regex, remoção de stopwords customizadas
e lematização através do spaCy para a língua portuguesa.
"""

import re
import spacy
from typing import List, Set, Optional


def obter_stopwords_customizadas(nlp_model: spacy.language.Language) -> Set[str]:
    """Retorna as stopwords do spaCy preservando negações críticas para análise de sentimento."""
    stopwords = nlp_model.Defaults.stop_words.copy()
    negacoes = {"não", "nao", "nem", "nunca", "jamais", "nada", "tampouco"}
    return stopwords - negacoes


def limpeza_basica_regex(texto: str) -> str:
    """Executa normalização preliminar de texto em lote com alta velocidade.

    - Converte para minúsculas
    - Remove tags HTML e URLs
    - Preserva apenas letras e acentuação da língua portuguesa
    - Normaliza espaços em branco duplicados
    """
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto = re.sub(r"<.*?>", " ", texto)
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def lematizar_documento(
    doc: spacy.tokens.Doc, stopwords_remover: Set[str], min_token_len: int = 2
) -> str:
    """Extrai lemas filtrando termos irrelevantes e palavras curtas."""
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stopwords_remover and len(token.text) >= min_token_len
    ]
    return " ".join(tokens)
