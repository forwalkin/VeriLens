# 0001 — Metodologia de avaliação do projeto-base

**Problema.** O README do projeto-base traz F1 de 0,69–0,90 sem script nem dados, e dois dos três
detectores foram calibrados com imagens sintéticas (ruído 1/f e grades senoidais). Não sabemos se
discriminam fotos reais de imagens de IA.

**Alternativas.** (a) confiar no README; (b) avaliar só com limiar 0.5 e F1; (c) avaliar com AUC +
IC e limiar 0.5, por grupo; (d) já recalibrar/ajustar pesos.

**Escolha.** (c), executando a implementação original sem alterações (commit `4ec7edc` e árvore limpa
verificados em runtime). Ajustes (d) ficam para uma etapa separada, em outros diretórios de resultados.

**Motivo.** O AUC com IC responde à pergunta de capacidade discriminativa sem depender de um limiar
que talvez esteja mal calibrado; o limiar 0.5 mostra o comportamento real do projeto como está.
A auditoria de confundidores evita elogiar um detector que só acerta por formato ou resolução.

**Critério de passagem (confirmado pelo responsável do projeto).** Limite inferior do IC95% do AUC > 0,60.
É um critério **operacional interno**, não um limite científico universal. Refinado pelo ADR 0002.
