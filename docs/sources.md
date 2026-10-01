# Fontes de dados

## Synthbuster (imagens de IA)
- Registro: https://zenodo.org/records/10066460 — DOI **10.5281/zenodo.10066460** (autor: Quentin Bammey, 2023).
- Arquivo: `synthbuster.zip`, 12,4 GB, md5 `0695bd328e16ea21c5c9cc2ae1d994ff` (conforme a página do registro).
- Conteúdo (conforme descrição do registro): 9 geradores — DALL·E 2, DALL·E 3, Adobe Firefly, Midjourney v5,
  Stable Diffusion 1.3, 1.4, 2, XL e GLIDE — com 1000 imagens por gerador; prompts e parâmetros em `prompts.csv`.
- As imagens são "vagamente baseadas" nas do RAISE-1k: para cada imagem do RAISE-1k gerou-se uma descrição
  (Midjourney /describe + CLIP interrogator), editada manualmente. O par compartilha tema, não é a mesma imagem.
- Os autores recomendam comparar as imagens sintéticas com as correspondentes do RAISE-1k e dizem que nenhuma
  imagem sofreu JPEG nem reamostragem (degradações devem ser adicionadas por nós, de forma controlada).
- Licença: **CC BY-NC-SA 4.0** (uso não comercial, atribuição, mesma licença em derivados). O dataset local e
  suas versões editadas ficam fora do git (`.gitignore`); não redistribuir.
- Observação: a página do registro cita outro DOI apenas como "related work" (supplement); ele não é o registro
  do dataset. Use sempre o DOI acima.
- **Estrutura interna do ZIP (pastas, nomes, relação com prompts.csv): AINDA NÃO VERIFICADA.** Use
  `scripts/inspect_synthbuster.py` e preencha a seção abaixo com o resultado.

## RAISE-1k (fotos reais)
- Download: http://loki.disi.unitn.it/RAISE/download.html (subconjunto RAISE-1k). Dataset destinado a pesquisa
  e educação não comercial (confirmado pelo responsável do projeto na página oficial).
- Formato/estrutura dos arquivos: a confirmar no download.

## Resultado da inspeção (preencher)
Pendente: executar `python scripts/inspect_synthbuster.py` e colar aqui `dataset/inspection/synthbuster_inspection.md`.
