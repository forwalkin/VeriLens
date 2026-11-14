# Fontes de dados

## Synthbuster (imagens de IA)
- Registro: https://zenodo.org/records/10066460 — DOI **10.5281/zenodo.10066460** (autor: Quentin Bammey, 2023).
- Arquivo: `synthbuster.zip`, 12,4 GB, md5 `0695bd328e16ea21c5c9cc2ae1d994ff` (conforme a página do registro).
- Conteúdo (conforme descrição do registro): 9 geradores — DALL·E 2, DALL·E 3, Adobe Firefly, Midjourney v5,
  Stable Diffusion 1.3, 1.4, 2, XL e GLIDE — com 1000 imagens por gerador; prompts e parâmetros em `prompts.csv`.
- As imagens são "vagamente baseadas" nas do RAISE-1k: para cada imagem do RAISE-1k gerou-se uma descrição
  (Midjourney /describe + CLIP interrogator), editada manualmente. Por isso usamos **pareamento por
  conteúdo/descrição**: o par compartilha tema, não é a mesma fotografia nem é pixel-pareado.
- Os autores recomendam comparar as imagens sintéticas com as correspondentes do RAISE-1k e dizem que nenhuma
  imagem sofreu JPEG nem reamostragem (degradações devem ser adicionadas por nós, de forma controlada).
- Licença: **CC BY-NC-SA 4.0** (uso não comercial, atribuição, mesma licença em derivados). O dataset local e
  suas versões editadas ficam fora do git (`.gitignore`); não redistribuir.
- Observação: a página do registro cita outro DOI apenas como "related work" (supplement); ele não é o registro
  do dataset. Use sempre o DOI acima.
- **Estrutura interna do ZIP: inspecionada em 14/11/2026** (ver "Resultado da inspeção" abaixo). O pareamento
  completo entre os nomes de arquivo e o `prompts.csv` ainda **NÃO está comprovado** (só uma amostra foi comparada).

## RAISE-1k (fotos reais)
- Download: http://loki.disi.unitn.it/RAISE/download.html (subconjunto RAISE-1k). Dataset destinado a pesquisa
  e educação não comercial (confirmado pelo responsável do projeto na página oficial).
- Formato/estrutura dos arquivos: a confirmar no download.

## Resultado da inspeção
Resumo revisado. Os arquivos brutos (`dataset/inspection/*.json|md`) ficam locais e fora do git, porque contêm
trechos do `prompts.csv`.

**Execução:** 14/11/2026, 21:00 UTC, pela Manus (Linux), na `main` `b39e6af`, com
`python scripts/inspect_synthbuster.py` em modo remoto (HTTP Range, sem `--zip`). Duração: 182,4 s. Resultado: sucesso.

**Transferência**
| item | valor |
|---|---|
| ZIP | 12.372.557.226 bytes (12,37 GB) |
| bytes recebidos | 24.949.423 (cerca de 0,2%; valor mínimo, pois os bytes parciais da tentativa com timeout não são contados) |
| requisições HTTP | 29 (inclui a sondagem e a tentativa repetida) |
| retentativas | 1: `TimeoutError` no intervalo 87842436-88891011; recuperada na repetição |
| intervalos incompletos | nenhum |

**Estrutura:** pasta raiz `synthbuster/`, com 9.007 entradas: 9.000 imagens (9 geradores x 1.000) e 7 arquivos
auxiliares (`readme.md`, `.directory`, `LICENCE.txt`, `prompts.csv`, `dalle2/prompts.txt`, `dalle2/generate.py`,
`dalle2/files.txt`). Os 9 geradores esperados foram encontrados, sem pastas extras. Todos os membros estão
comprimidos individualmente em `deflate`, então dá para ler só os escolhidos por Range (validado nas amostras).

| pasta | imagens | MB médios | amostra de 3 arquivos (dimensões, formato real) |
|---|---|---|---|
| dalle2 | 1000 | 3,15 | 1024x1024 PNG RGB |
| dalle3 | 1000 | 2,16 | 1024x1024 e 1792x1024, PNG **RGBA** |
| firefly | 1000 | 1,55 | 2048x2048 e 2304x1792, conteúdo **JPEG** (extensão `.png`) |
| glide | 1000 | 0,09 | 256x256 PNG RGB |
| midjourney-v5 | 1000 | 1,94 | 1360x896 PNG RGB |
| stable-diffusion-1-3 | 1000 | 0,54 | 512x512 PNG RGB |
| stable-diffusion-1-4 | 1000 | 0,54 | 512x512 PNG RGB |
| stable-diffusion-2 | 1000 | 1,95 | 896x1152, 1344x704 e 832x1152, PNG RGB |
| stable-diffusion-xl | 1000 | 1,66 | 896x1152, 1344x704 e 832x1152, PNG RGB |

**`synthbuster/prompts.csv`:** 873.324 bytes, 1.000 linhas, 33 colunas. A primeira, `image name (matching Raise-1k)`,
tem 1.000 valores distintos (formato `r0b0e2ba5t`) e é a chave candidata para o pareamento; os parâmetros das demais
colunas são específicos de cada gerador. Os 8 nomes de arquivo comparados (5 primeiros e 3 últimos, em ordem
alfabética) são exatamente os mesmos nas 9 pastas, e os 8 casam com essa coluna.

**Pareamento (não comprovado):** os 5 padrões de nome mais comuns e as suas contagens são idênticos nas 9 pastas, o
que é coerente com os mesmos nomes de arquivo em todas elas. Mas esses 5 padrões cobrem só 137 dos 1.000 nomes por
pasta, e a comparação com o `prompts.csv` usou só 8 nomes por pasta (os mesmos em todas). A comparação completa
(todos os nomes de cada pasta contra as 1.000 chaves) fica para a etapa 2.

**Pontos para a auditoria de confundidores (etapa 5):** observados em apenas 3 arquivos por pasta, não em todas as
imagens.
1. A resolução varia muito entre geradores (de 256x256 no GLIDE a 2048x2048 no Firefly) e também dentro do
   DALL·E 3, SD 2 e SDXL. Um detector poderia separar por tamanho e não por geração.
2. No Firefly, 3 de 3 amostras são JPEG apesar da extensão `.png`. A descrição do registro diz que nenhuma imagem
   sofreu JPEG, então isso precisa ser verificado em todas as imagens.
3. O DALL·E 3 vem em modo RGBA (canal alfa).
4. O formato e a estrutura dos arquivos do RAISE-1k ainda estão por confirmar.

**Ainda não lidos:** `LICENCE.txt` (216 bytes) e `readme.md` (1.838 bytes) de dentro do ZIP.
