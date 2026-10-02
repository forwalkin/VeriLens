# ai-scanner-eval — benchmark reproduzível do projeto-base

Mede se os detectores do `RitikPatill/ai-image-authenticity-scanner` (commit `4ec7edc`)
**separam fotos reais de imagens de IA**, antes de qualquer port para Android.
Roda a implementação **original, sem alterações**. Experimentos de melhoria ficam
em outra etapa, com resultados em pastas separadas (ver `docs/decisions/0001`).

## Pergunta que o benchmark responde

Frequency, CLIP e SDXL Detector distinguem real de IA melhor que o acaso?
Resposta por detector via **AUC com IC 95%** (independe de limiar), no global **e por gerador**, com n
por gerador. Também reportamos o que o projeto original reportaria (limiar 0.5): accuracy, precision,
recall, F1, matriz de confusão. O relatório termina em "Decisão de port para Android" (preliminar).

**Critério interno (não é limite científico universal):** limite inferior do IC95% do AUC > 0,60 — global,
em cada gerador com n ≥ 30, nos grupos de edição, e sem sinal de confundidor. Um AUC global alto não basta
se algum gerador relevante estiver próximo do acaso. Frequency e CLIP **não são descartados** só pelo global:
a regra separa "manter para investigação / recalibrar / descartar" com as evidências listadas (ADR 0002).

## Fluxo definitivo (siga a ordem; não pule etapas)

```
 1. Inspecionar Synthbuster
        ↓
 2. Confirmar estrutura/chaves
        ↓
 3. Configurar sources.toml
        ↓
 4. Montar dataset
        ↓
 5. Auditar confundidores
        ↓
 6. Criar edited / ai_edited
        ↓
 7. Benchmark pequeno com modelos reais
        ↓
 8. sanity_check
        ↓
 9. Benchmark completo
        ↓
10. report.md + AUDIT.md
        ↓
11. Decisão sobre quais detectores merecem port Android
```

O benchmark pequeno (7) **sempre** roda antes do completo (9), e o completo só roda se o `sanity_check` (8) passar.

```bash
pip install -r requirements.txt
bash scripts/setup_base.sh                          # base no commit 4ec7edc + dependências (torch etc.)

# 1) Inspecionar o Synthbuster SEM baixar os 12,4 GB (lê só o índice do ZIP por HTTP Range)
python scripts/inspect_synthbuster.py               # -> dataset/inspection/synthbuster_inspection.{json,md}

# 2) Confirmar estrutura/chaves: pastas, nomes de arquivo, identificadores, relação com prompts.csv,
#    possibilidade de seleção parcial. Cole o resultado em docs/sources.md.
#    Se os nomes dos arquivos não casarem com os do RAISE-1k, prepare o map_csv (ai_key,real_key).

# 3) Configurar fontes
cp sources.example.toml sources.toml                # ajuste caminhos, [pairing] e map_csv

# 4) Montar o dataset (só os arquivos selecionados são transferidos; kind="remote_zip")
python scripts/build_dataset.py --sources sources.toml     # grava manifest.csv e pairing_summary.json

# 5) Auditar confundidores (formato, resolução, EXIF)
python scripts/audit_dataset.py                     # -> dataset/AUDIT.md

# 6) Criar edited (real editada) e ai_edited (IA editada)
python scripts/make_edited.py --per-edit 30

# 7) Benchmark PEQUENO com os modelos reais (não é resultado: n minúsculo)
python scripts/run_benchmark.py --sample-per-group 2 --run-id validation

# 8) Sanidade: sem erros, scores em [0,1], ensemble == média ponderada original, determinismo
python scripts/sanity_check.py results/validation

# 9) Benchmark COMPLETO (somente se o passo 8 passou)
python scripts/run_benchmark.py --run-id full
python scripts/sanity_check.py results/full

# 10) Relatório (+ dataset/AUDIT.md do passo 5)
python scripts/report.py results/full               # report.md + score_distributions.png

# 11) Decisão: ler a seção "Decisão de port para Android" do report.md e AUDIT.md; a decisão final é nossa.
pytest -q
```

`--sample-per-group N` seleciona até N imagens por (rótulo, gerador|edição) com seed fixa.

## Dataset

`dataset/real`, `dataset/ai`, `dataset/edited`, `dataset/ai_edited` + `dataset/manifest.csv` (id = sha256[:16],
rótulo, fonte, gerador, licença, pai e edição das editadas, `pair_key`). Bytes originais preservados.

**Meta de pares.** Para julgar cada um dos 9 geradores individualmente (n ≥ 30 por gerador) são necessários no
mínimo **9 × 30 = 270 pares**. Alvo inicial: **300 pares** (`n_pairs = 300`), ou seja, 300 imagens do RAISE-1k +
300 × 9 imagens do Synthbuster = **3000 imagens**, antes das versões editadas. O Synthbuster tem ~1,4 MB por imagem
em média (12,4 GB / 9000; estimativa), então as 2700 imagens somam da ordem de 3,7 GB.
`n_pairs` é um **máximo desejado, não uma obrigação**: se houver menos chaves realmente comuns, usa-se o máximo
disponível e o `report.md` informa pares solicitados × chaves comuns disponíveis × pares usados (e avisa que
geradores com n < 30 não são julgados). Edições: 17 × 30 por classe; por edição individual os ICs são largos,
use os grupos (compressão, geometria, tonal).

Fontes (ver `docs/sources.md`; conferir licença e estrutura antes de usar):
- **IA:** Synthbuster (https://zenodo.org/records/10066460, DOI 10.5281/zenodo.10066460, CC BY-NC-SA 4.0):
  9 geradores × 1000 imagens, ~12,4 GB.
- **Real:** RAISE-1k (comparação recomendada pelos autores do Synthbuster), fotos reais adicionais quando disponíveis
  (COCO val2017, **fotos suas** de câmeras/celulares variados).
- **Pareamento por conteúdo/descrição:** `[pairing]` em `sources.toml` escolhe UM conjunto de chaves (nome de
  arquivo sem extensão, ou `map_csv` com `ai_key,real_key`) presente no real e em todos os geradores. O Synthbuster
  foi gerado a partir de *descrições* das imagens do RAISE-1k: o par compartilha tema/descrição, **não é a mesma
  fotografia** e não há correspondência pixel a pixel. A nomenclatura real dos arquivos só é conhecida após a
  inspeção (passo 1). Confira formato e tamanho dos arquivos do RAISE-1k na página do download.
- **Edited / ai_edited:** `make_edited.py` aplica 17 edições determinísticas a reais (`edited`) e a IA
  (`ai_edited`, pais em rodízio entre geradores). Só com real editada mediríamos falsos positivos; a IA editada
  mede se o sinal sobrevive a compressão/edição. `screenshot_sim` é simulação: adicione também capturas/fotos
  de tela reais manualmente.

**Vieses:** real e IA costumam diferir em formato, resolução e EXIF. O `audit_dataset.py` (passo 5) marca como
confundidores os atributos com AUC ≥ 0,8 ou ≤ 0,2; eles devem constar na interpretação. Possível sobreposição entre
o treino do `Organika/sdxl-detector` e as fontes é desconhecida; trate o SDXL Detector como possivelmente otimista
em imagens SDXL.

## Convenções

- Positivo = IA (`ai`, `ai_edited`). Negativo = `real`, `edited`. IA parcialmente editada está fora do escopo.
- Comparação principal: REAL vs IA. Robustez: real editada vs IA editada com a mesma edição; os grupos de
  edição usam bootstrap por imagem-pai (pais repetem entre edições).
- Limiar 0.5 e pesos 0.30/0.35/0.35 são os do projeto original e não são ajustados aqui.
- Cada execução grava `meta.json` (commit, hash do manifesto, hash da calibração, versões, CPU).

## Licenças

Projeto-base: MIT. `Organika/sdxl-detector`: CC BY-NC 3.0 (uso não comercial, com atribuição).
Synthbuster: CC BY-NC-SA 4.0 (não comercial, mesma licença em derivados; não redistribuir o dataset/edições).
RAISE-1k: pesquisa e educação não comercial. Respeite as licenças de cada fonte de imagens.

## Estado do repositório / permissões

`scripts/setup_base.sh` está versionado como `100755` (um teste em `tests/test_repo_hygiene.py` falha se isso
mudar). Se um zip/sistema de arquivos descartar o bit de execução, o git mostrará `old mode 100755 / new mode 100644`:
restaure com `git checkout -- scripts/setup_base.sh` (ou `git update-index --chmod=+x scripts/setup_base.sh`)
ou apenas rode `bash scripts/setup_base.sh`, que não exige o bit. Prefira o `.bundle` ao zip: ele preserva o histórico e os modos.
