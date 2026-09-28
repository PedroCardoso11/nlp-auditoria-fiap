# 📊 Auditoria de NLP - Análise de Sentimentos em Avaliações de Clientes (B2W)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![spaCy](https://img.shields.io/badge/spaCy-pt__core__news__sm-09a3d5.svg)](https://spacy.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F1--Macro-F7931E.svg)](https://scikit-learn.org/)
[![HuggingFace BERTimbau](https://img.shields.io/badge/HuggingFace-BERTimbau-yellow.svg)](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
[![Version](https://img.shields.io/badge/version-v1.0.0-green.svg)](https://github.com/)

Projeto acadêmico e pipeline prático de **Processamento de Linguagem Natural (NLP)** e **Machine Learning** desenvolvido para auditoria de maturidade em sistemas de IA, aplicado sobre o corpus **B2W-Reviews01** contendo mais de **132 mil avaliações reais de clientes de e-commerce brasileiro**.

O projeto avalia rigorosamente a aplicação nas **6 Dimensões Fundamentais de Maturidade de NLP**, gerando a tabela de conformidade, o gráfico radar de competências, o plano de ação priorizado e as respostas reflexivas para o fórum da disciplina.

---

## 🏗️ Estrutura do Projeto

A organização de diretórios segue as melhores práticas de Engenharia de Software e *Data Science*:

```text
├── .gitignore               # Arquivos e diretórios ignorados pelo Git
├── README.md                # Documentação executiva do repositório
├── requirements.txt         # Dependências do projeto para ambiente virtual
├── data/
│   ├── README.md            # Dicionário de dados e instruções do dataset
│   ├── raw/                 # Dados brutos imutáveis
│   │   └── B2W-Reviews01.csv
│   └── processed/           # Dados processados e intermediários
├── docs/
│   └── images/              # Diretrizes do exercício e capturas de tela da auditoria
│       ├── 01_dimensoes_maturidade_nlp.png
│       ├── 02_escala_avaliacao_auditoria.png
│       ├── 03_gaps_e_oportunidades.png
│       ├── 04_plano_de_acao.png
│       └── 05_questoes_para_discussao.png
├── notebooks/
│   └── Exercicio_aula1_fiap.ipynb   # Notebook principal documentado e validado
└── src/
    ├── __init__.py          # Inicializador do pacote
    └── preprocessing.py     # Módulo reutilizável de limpeza e lematização
```

---

## 🔍 As 6 Dimensões de Maturidade Avaliadas

O pipeline cobre de ponta a ponta o ciclo de vida analítico e de modelagem:

1. **Fundamentos e Definição do Problema:**
   * Corpus de 132k registros B2W;
   * Framing analítico com 3 tarefas (Classificação multiclasse 1 a 5 estrelas, NER e Modelagem de Tópicos);
   * Tratamento explícito de dados nulos e definição do **F1-Score Macro** como métrica primária em função do desbalanceamento severo.
2. **Pré-processamento e Limpeza Otimizada:**
   * Higienização com expressões regulares (remoção de HTML, links e ruídos);
   * Preservação inteligente de termos de negação (*"não"*, *"nunca"*, *"jamais"*) na lista de stopwords;
   * Lematização em português via `pt_core_news_sm` acelerada em lote paralelo com `nlp.pipe()`.
3. **Vetorização Clássica (TF-IDF):**
   * Extração de unigramas e bigramas (`ngram_range=(1,2)`) com corte de frequência e atenuação logarítmica (`sublinear_tf=True`);
   * Classificador baseline interpretável (Regressão Logística) com extração dos coeficientes das palavras mais determinantes para insatisfação (1 estrela) e satisfação (5 estrelas).
4. **Embeddings Semânticos:**
   * Treinamento contínuo de **Word2Vec** com teste de similaridade do cosseno e vizinhos mais próximos;
   * Extração de representações contextuais profundas densas (**BERTimbau** - token `[CLS]`) e projeção bidimensional via PCA.
5. **Modelagem Supervisionada e Validação Cruzada:**
   * Validação Cruzada Estratificada com 5 Folds comparando Multinomial Naive Bayes, Regressão Logística e LinearSVC;
   * Mitigação do viés de desbalanceamento aplicando **Pesos de Classe** (`class_weight='balanced'`), com demonstração do ganho substancial em F1-Macro.
6. **Análise Não-Supervisionada e Interpretação:**
   * Reconhecimento de Entidades Nomeadas (NER) para identificação de marcas comerciais e fabricantes;
   * Modelagem de tópicos latentes com **Latent Dirichlet Allocation (LDA)** e visualização espacial via **t-SNE**, gerando 4 pilares de insights acionáveis de negócio (logística, produto, pós-venda e atendimento).

---

## 📋 Resultados da Auditoria Formal (Passos 1, 2 e 3)

### Passo 1: Avaliação por Dimensão (Escala Oficial)
* **N - Não Atendido** | **P - Parcialmente Atendido** | **L - Largamente Atendido** | **F - Totalmente Atendido**

| Dimensão Fundamental | Nota | Status no Projeto |
| :--- | :---: | :--- |
| **1. Fundamentos e Definição** | **F** | Corpus de 132k tratado, nulos removidos, framing e métricas formais estabelecidos. |
| **2. Pré-processamento** | **F** | spaCy paralelo via `nlp.pipe()`, lematização e stopwords preservando negações. |
| **3. Vetorização Clássica** | **F** | TF-IDF com n-grams (1,2) e interpretabilidade analítica dos coeficientes. |
| **4. Embeddings Semânticos** | **L** | Word2Vec local treinado e BERTimbau avaliado via PCA em amostras (limite de GPU). |
| **5. Modelagem Supervisionada** | **F** | Validação cruzada estratificada e mitigação de viés via pesos de classe. |
| **6. Análise Não-Supervisionada** | **F** | NER comercial e LDA 4 tópicos com projeção t-SNE e insights de negócio. |

### Passo 2: Radar de Maturidade e Gaps
O notebook gera visualmente o gráfico radar comparando a maturidade atual com o nível máximo de excelência (**Nível F**):
* **Gap 1 (Hardware/GPU):** Necessidade de aceleradores para inferência em lote com BERTimbau.
* **Gap 2 (Vocabulário OOV):** Tratamento de termos de internet e erros ortográficos com FastText/n-grams de caracteres.
* **Gap 3 (Ambiguidade Neutra):** Baixo recall das notas 2 e 3 devido à dispersão semântica das opiniões mistas.

### Passo 3: Plano de Ação Estruturado
* **Quick Wins (Curto Prazo - 1 a 2 semanas):** Descarte de nulos, ativação obrigatória de `class_weight='balanced'` e pré-processamento paralelo com `nlp.pipe()`. *(Ganho imediato de F1 com custo zero)*.
* **Médio Prazo (1 a 2 meses):** FastText com sub-palavras contra erros ortográficos, tuning via GridSearchCV e score de coerência $C_v$ no LDA. *(Aumento de 5% de acurácia em textos informais)*.
* **Longo Prazo (3 a 6 meses):** Fine-tuning supervisionado de BERTimbau em GPU, migração para BERTopic e quantização ONNX (INT8) para microsserviço de baixa latência (< 50ms).

---

## 🚀 Como Executar o Projeto Localmente

### 1. Clonar o Repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Criar e Ativar o Ambiente Virtual
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependências e Modelo do spaCy
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

### 4. Executar o Jupyter Notebook
```bash
jupyter notebook notebooks/Exercicio_aula1_fiap.ipynb
```

---

## 👥 Versionamento e Licença
* **Versão:** `v1.0.0`
* **Padrão de Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
* **Autor:** Pedro Cardoso (pedrocardosopsn@gmail.com)
