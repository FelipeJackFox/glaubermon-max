> **Evaluación del candidato recibido:** seguir [RECUPERAR_EVALUACION.md](docs/RECUPERAR_EVALUACION.md) para validar la optimización del encoder y la búsqueda por lotes en CPU/CUDA y comparar los pesos congelados antes de entrenar más.

> **Piloto habilitado para los cinco equipos validados.** Entrega con los comandos completos para NVIDIA: [docs/ENTRENAR_GPU.md](docs/ENTRENAR_GPU.md). La validación local fue en CPU; se recibió un candidato entrenado de 100 partidas y falta completar su evaluación. Ver [estado y pruebas](docs/alignment/ESTADO.md).

> Estado de las correcciones y comandos del piloto oficial: [docs/TRAINING_CORRECTIONS.md](docs/TRAINING_CORRECTIONS.md). El entrenamiento actual usa observaciones del cliente y resultados de Showdown; el motor interno sigue siendo experimental. Los checkpoints originales se conservan.

# Glaubermon Max

Glaubermon Max is an expert/superhuman game-theoretic AI for **Pokémon Showdown**.

## Key Innovations
1. **Bayesian Particle Filter & Belief State Engine**: Accurately tracks hidden moves, abilities, items, EV spreads, and Tera types.
2. **Exact Reverse Damage Inversion**: Deduces stat benchmarks and boosts from damage rolls.
3. **Simultaneous-Move Nash Matrix Solver**: Solves normal-form matrix games via Linear Programming to output unexploitable mixed strategies ($\pi^*$).
4. **Permutation-Invariant Set Transformer**: Multi-head self- and cross-attention representation for 6v6 team structures.
5. **AlphaStar-Style League Training**: Multi-agent population-based training with Prioritized Fictitious Self-Play.
