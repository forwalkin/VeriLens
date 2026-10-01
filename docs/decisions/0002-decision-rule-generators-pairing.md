# 0002 — Regra de decisão preliminar, análise por gerador, pareamento e IA editada

**Problema.** Um AUC global alto pode esconder um gerador onde o detector está no acaso, depender de
formato/resolução, ou desaparecer após compressão. O responsável também pediu que Frequency e CLIP não
sejam descartados apenas pelo resultado global.

**Alternativas.** (a) decidir só pelo AUC global; (b) regra por evidência: global + por gerador + estratos de
confundidor + robustez; (c) decisão 100% manual sem regra prévia.

**Escolha.** (b), com a regra fixada em `evalkit/decision.py` antes de ver resultados:
- critério interno: limite inferior do IC95% do AUC > 0,60;
- por gerador: só julga com n ≥ 30 por classe; "próximo do acaso" = IC contém 0,5 (ou invertido);
- confundidor: atributo de metadado com AUC ≥ 0,8 ou ≤ 0,2 exige estrato correspondente (formato ou resolução)
  onde o detector mantenha o critério; queda > 0,15 de AUC num estrato é lacuna; `has_exif`, `bytes_per_pixel`
  e `aspect_ratio` não são controláveis por estrato e, se sinalizados, ficam como lacuna;
- robustez: AUC (real editada vs IA editada) por grupo de edição, bootstrap por imagem-pai;
- saída: *manter para investigação* (sem lacunas) / *recalibrar* (lacunas, mas há evidência de sinal) /
  *descartar* (global não passa e nenhum gerador julgado passa) / *dados insuficientes*.
O relatório mostra todas as evidências e a lista de lacunas; a decisão final continua sendo nossa.

**Pareamento.** `[pairing]` escolhe um conjunto único de chaves comum ao RAISE-1k e a todos os geradores do
Synthbuster; o relatório calcula a taxa de vitória pareada (P(IA > real no mesmo par)) com IC por bootstrap
de pares. Reduz confundimento de conteúdo, sem eliminar o de formato/resolução (Synthbuster = PNG, RAISE =
TIFF/RAW de alta resolução). Os pares compartilham tema, não são a mesma imagem.

**IA editada (adição de escopo, justificada).** Com apenas real editada só medimos falso positivo; o pedido de
avaliar "comportamento após compressão/edição" exige também o efeito sobre IA. Gerada por `make_edited.py`
(`ai_edited`), com pais em rodízio entre geradores.

**Limitações.** Reais compartilhadas entre geradores tornam os ICs por gerador correlacionados entre si; ICs por
edição são largos; heurísticas de confundidor são univariadas e não provam causalidade.
