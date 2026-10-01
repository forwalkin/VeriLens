# ai-scanner-eval — benchmark reproduzível do projeto-base

Mede se os detectores do `RitikPatill/ai-image-authenticity-scanner` (commit `4ec7edc`)
**separam fotos reais de imagens de IA**, antes de qualquer port para Android.
Roda a implementação **original, sem alterações**. Experimentos de melhoria ficam
em outra etapa, com resultados em pastas separadas (ver `docs/decisions/0001`).

## Pergunta que o benchmark responde

Frequency, CLIP e SDXL Detector distinguem real de IA melhor que o acaso?
Resposta por detector via **AUC com IC 95%** (independe de limiar). Se o IC contém 0.5,
o detector é "indistinguível do acaso". Também reportamos o que o projeto original reportaria
(limiar 0.5): accuracy, precision, recall, F1, matriz de confusão.

## Uso

```bash
pip install -r requirements.txt
scripts/setup_base.sh                      # clona o base no commit fixado e instala (torch etc.)
cp sources.example.toml sources.toml       # ajuste caminhos/fontes (veja avisos no arquivo)
python scripts/build_dataset.py --sources sources.toml
python scripts/make_edited.py --per-edit 20
python scripts/audit_dataset.py            # confundidores: formato, resolução, EXIF
python scripts/run_benchmark.py            # exige base limpa no commit fixado
python scripts/report.py results/<run_id>  # report.md + score_distributions.png
pytest -q                                  # testes de métricas, edições e do encanamento
```

## Dataset

`dataset/real`, `dataset/ai`, `dataset/edited` + `dataset/manifest.csv` (id = sha256[:16],
rótulo, fonte, gerador, licença, pai e edição das editadas). Bytes originais preservados.

Tamanho mínimo sugerido: ≥300 reais, ≥300 IA (≥30 por gerador), 17 edições × 20 = 340 editadas.
Com ~300 por classe o IC do AUC fica por volta de ±0,03; por gerador (n≈40) a taxa tem IC de ±0,15 —
diferenças pequenas entre geradores não são conclusivas.

Fontes sugeridas (conferir licença e estrutura antes de usar):
- **IA:** Synthbuster (Zenodo, 10.5281/zenodo.10066047): DALL·E 2/3, Firefly, Midjourney v5, SD 1.3/1.4/2/XL, GLIDE.
- **Real:** RAISE-1k (par recomendado do Synthbuster), COCO val2017 e, principalmente, **fotos suas**
  de câmeras/celulares variados.
- **Edited:** geradas de `real` por `make_edited.py` (17 edições determinísticas: JPEG q75/50/30, resize,
  crop, cor, filtros, pipeline "social", `screenshot_sim`). `screenshot_sim` é simulação: adicione também
  capturas/fotos de tela reais como `edited` manualmente.

**Vieses:** real e IA costumam diferir em formato (JPEG vs PNG), resolução e EXIF. Rode
`audit_dataset.py` primeiro: atributos com AUC ≥0,8 ou ≤0,2 são confundidores e devem constar
na interpretação. Possível sobreposição entre o treino do `Organika/sdxl-detector` e as fontes
é desconhecida; trate o SDXL Detector como possivelmente otimista em imagens SDXL.

## Convenções

- Positivo = IA. REAL e EDITADA (real editada) = negativos. IA parcialmente editada está fora do escopo.
- Comparação principal: REAL vs IA (independentes). EDITADA é reportada à parte (taxa de falso positivo
  por edição), porque pais repetidos inflariam a confiança do IC.
- Limiar 0.5 e pesos 0.30/0.35/0.35 são os do projeto original e não são ajustados aqui.
- Cada execução grava `meta.json` (commit, hash do manifesto, hash da calibração, versões, CPU).

## Licenças

Projeto-base: MIT. `Organika/sdxl-detector`: CC BY-NC 3.0 (uso não comercial, com atribuição).
Respeite as licenças de cada fonte de imagens.
