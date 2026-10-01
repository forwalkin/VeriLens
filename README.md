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

## Uso (em etapas; não pule a inspeção)

```bash
pip install -r requirements.txt
bash scripts/setup_base.sh                         # base no commit 4ec7edc + dependências (torch etc.)

# 1) INSPEÇÃO do Synthbuster sem baixar os 12,4 GB (lê só o índice do ZIP por HTTP Range)
python scripts/inspect_synthbuster.py              # -> dataset/inspection/synthbuster_inspection.{json,md}
#    Confira: pastas, nomes de arquivo, identificadores, relação com prompts.csv, seleção parcial.
#    Cole o resultado em docs/sources.md e ajuste [pairing]/map_csv se os nomes não casarem com o RAISE-1k.

# 2) Dataset (só os arquivos selecionados são transferidos; kind="remote_zip")
cp sources.example.toml sources.toml               # ajuste caminhos/fontes
python scripts/build_dataset.py --sources sources.toml
python scripts/make_edited.py --per-edit 30        # real editada + IA editada (robustez)
python scripts/audit_dataset.py                    # confundidores: formato, resolução, EXIF

# 3) Benchmark PEQUENO de validação (verifica o pipeline com os modelos reais; ~poucas dezenas de imagens)
python scripts/run_benchmark.py --sample-per-group 2 --run-id validation
python scripts/sanity_check.py results/validation  # sem erros, scores em [0,1], ensemble == média ponderada, determinismo
python scripts/report.py results/validation        # NÃO interpretar como resultado: n minúsculo

# 4) Benchmark COMPLETO, só se a validação estiver coerente
python scripts/run_benchmark.py
python scripts/sanity_check.py results/<run_id>
python scripts/report.py results/<run_id>          # report.md + score_distributions.png
pytest -q
```

`--sample-per-group N` seleciona até N imagens por (rótulo, gerador|edição) com seed fixa. `--run-id` evita
misturar a validação com o run completo.

## Dataset

`dataset/real`, `dataset/ai`, `dataset/edited` + `dataset/manifest.csv` (id = sha256[:16],
rótulo, fonte, gerador, licença, pai e edição das editadas). Bytes originais preservados.

Tamanho sugerido: ≥100 pares Synthbuster↔RAISE (mesmas chaves em todos os geradores; ≥30 por gerador é o
mínimo para julgar), mais fotos reais próprias; edições: 17 × 30 por classe. Por gerador (n≈100) o IC do
AUC fica em torno de ±0,05; com n=30, ±0,10 — diferenças pequenas entre geradores não são conclusivas.
Por edição individual (n=30 por classe) os ICs são largos: use os grupos (compressão, geometria, tonal).

Fontes sugeridas (conferir licença e estrutura antes de usar):
- **IA:** Synthbuster (https://zenodo.org/records/10066460, DOI 10.5281/zenodo.10066460, CC BY-NC-SA 4.0):
  9 geradores × 1000 imagens, ~12,4 GB. Ver `docs/sources.md`.
- **Real:** RAISE-1k (par recomendado do Synthbuster), COCO val2017 e, principalmente, **fotos suas**
  de câmeras/celulares variados.
- **Pareamento:** `[pairing]` em `sources.toml` escolhe UM conjunto de chaves (nome de arquivo sem extensão,
  ou `map_csv` com `ai_key,real_key`) presente no real e em todos os geradores. O Synthbuster foi gerado a partir
  de *descrições* do RAISE: o par compartilha tema, não é a mesma imagem. A nomenclatura real dos arquivos
  não foi verificada por mim. RAISE são imagens de dezenas de MP: espere centenas de MB por TIFF.
- **Edited / ai_edited:** `make_edited.py` aplica 17 edições determinísticas a reais (`edited`) e a IA
  (`ai_edited`, pais em rodízio entre geradores). Só com real editada mediríamos falsos positivos; a IA editada
  mede se o sinal sobrevive a compressão/edição. `screenshot_sim` é simulação: adicione também capturas/fotos
  de tela reais manualmente.

**Vieses:** real e IA costumam diferir em formato (JPEG vs PNG), resolução e EXIF. Rode
`audit_dataset.py` primeiro: atributos com AUC ≥0,8 ou ≤0,2 são confundidores e devem constar
na interpretação. Possível sobreposição entre o treino do `Organika/sdxl-detector` e as fontes
é desconhecida; trate o SDXL Detector como possivelmente otimista em imagens SDXL.

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
