# 📂 Diretório de Dados

Este diretório armazena os dados brutos e processados do projeto de Auditoria de NLP.

## 📄 Dataset: B2W-Reviews01.csv
* **Origem:** Repositório público da B2W Digital contendo avaliações reais de clientes em plataformas de e-commerce brasileiro (Americanas, Submarino, Shoptime).
* **Volume:** 132.373 registros e 14 colunas.
* **Tamanho do Arquivo:** ~47,2 MB.

### Dicionário de Dados Principal

| Coluna Original | Coluna Renomeada (PT) | Tipo | Descrição |
| :--- | :--- | :--- | :--- |
| `submission_date` | `data_envio` | String/Data | Data e hora de submissão do feedback |
| `reviewer_id` | `id_avaliador` | String | Identificador anônimo do usuário |
| `product_id` | `id_produto` | String | Código SKU do produto avaliado |
| `product_name` | `nome_produto` | String | Título cadastrado do item |
| `product_brand` | `marca_produto` | String | Marca ou fabricante |
| `site_category_lv1` | `categoria_site_nivel1` | String | Categoria macro no e-commerce |
| `review_title` | `titulo_avaliacao` | String | Título resumido da avaliação |
| `overall_rating` | `avaliacao_geral` | Inteiro | **Rótulo alvo**: nota de 1 a 5 estrelas |
| `recommend_to_a_friend` | `recomendar_amigo` | String (Yes/No) | Indicador binário de recomendação |
| `review_text` | `texto_avaliacao` | String | **Feature textual primária** analisada no NLP |
| `reviewer_state` | `estado_avaliador` | String | Sigla do estado federativo (UF) |

## Estrutura de Pastas
* `data/raw/`: Contém os arquivos brutos imutáveis (`B2W-Reviews01.csv`).
* `data/processed/`: Destinado a armazenar datasets transformados ou vetores intermediários serializados.
