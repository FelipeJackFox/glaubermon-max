# Recuperar la evaluación: medir CPU/CUDA antes de repetir cien partidas

Los archivos recibidos el 14 de septiembre confirman cien partidas de entrenamiento con 60 s por decisión. El candidato entregado tiene SHA-256 `3a4717efe548bca8514c649dd63175ed500f93ec7bdd9d4e053471b23ed10e0e`. No volver a entrenar ni sustituir esos pesos para este diagnóstico.

Las evaluaciones usaron profundidad 2, un proceso y 30 s por decisión. El original acumuló 81 timeouts y el candidato 99. La diferencia emparejada de puntuación con reloj fue −13 puntos porcentuales (IC por bloques −21 a −6): es un peor resultado operativo del candidato, pero los abandonos impiden atribuirlo exclusivamente a sus decisiones de juego. Los ZIP de ambos pilotos son duplicados del mismo experimento de 60 s.

## Corrección posterior al perfil CUDA: encoder directo

El perfil recibido confirma miles de estados codificados por decisión. Se eliminan las conversiones intermedias tensor→NumPy y los clones temporales por movimiento; se escriben las características directamente en buffers propios. El caché de movimientos conserva un máximo de 4,096 entradas y sus arrays internos son de solo lectura. Las funciones públicas siguen devolviendo tensores independientes. Se conserva `public_callbacks_v7`, la búsqueda y los checkpoints.

Los dos frames originales bajaron de 15.12/14.86 s a 11.41/11.06 s en CPU/macOS, con exactamente las mismas acciones, estrategia, valor y llamadas a la red. Pasaron las 587 pruebas. Las cinco trayectorias (297 estados) conservan características bit a bit. Son mediciones locales, no garantía de cumplir 30 s en RTX. [Evidencia](alignment/direct-encoding-results.json).

**Ejecutar ahora esta comprobación corta en RTX**, conservando la entrega anterior:

```sh
git pull --ff-only origin codex/simulator-training-corrections
python -u -m glaubermon.evaluation.diagnose_timeouts --benchmark runs/recovery-v7-01/smoke-original --checkpoint checkpoints/glaubermon_rebel_latest.pt --device cuda --output runs/diagnosis-direct-encoding-v7-01
python -c "import shutil; shutil.make_archive('entrega-diagnosis-direct-encoding-v7-01','zip',root_dir='runs',base_dir='diagnosis-direct-encoding-v7-01')"
```

Compartir el ZIP; no iniciar entrenamiento ni la evaluación larga todavía. Si sigue excediendo el reloj, esta optimización no basta y se debe continuar con el coste restante. La poda sigue sin activarse. Las secciones siguientes documentan el diagnóstico anterior.

## Resultado recibido: detener la evaluación larga

La entrega RTX del 14 de septiembre, commit `fbdf4e3` sin cambios locales, se detuvo con `smoke_failed`. CUDA ganó las sondas (31.98/5.53 s frente a 61.66/17.50 s en CPU), pero el original excedió 30 s en los turnos 17 y 12. Los controles heurísticos terminaron. No se evaluó el candidato en partidas ni se inició la etapa de 400 partidas. [Evidencia](alignment/rtx-recovery-results.json).

**Siguiente paso: solo diagnosticar los dos fallos en la RTX.** Mantener los pesos, profundidad y límite de evaluación originales. Este comando reproduce ambos frames primero sin profiler y luego con cProfile, guarda los perfiles y termina sin lanzar partidas ni entrenamiento:

```sh
git pull --ff-only origin codex/simulator-training-corrections
python -u -m glaubermon.evaluation.diagnose_timeouts --benchmark runs/recovery-v7-01/smoke-original --checkpoint checkpoints/glaubermon_rebel_latest.pt --device cuda --output runs/diagnosis-recovery-v7-01
python -c "import shutil; shutil.make_archive('entrega-diagnosis-recovery-v7-01','zip',root_dir='runs',base_dir='diagnosis-recovery-v7-01')"
```

Enviar ese ZIP. La carpeta de salida debe ser nueva. `--process-seconds` limita el proceso de diagnóstico (300 s por ejecución); no altera el reloj permitido al bot. Los tiempos con profiler tienen sobrecoste y no deben compararse como latencia de juego. Si una reproducción falla, se registra en `diagnosis.json`; se conservan las mediciones restantes.

El diagnóstico completo pasó localmente: ambos frames tardaron 15.12 y 14.86 s sin profiler; las ejecuciones con profiler conservaron exactamente acciones, estrategia y valor. Pasaron las dos regresiones de la nueva herramienta. [Evidencia local](alignment/timeout-diagnosis-local.json).

En macOS, los perfiles de ambos frames muestran miles de llamadas a la red además de simulación, encoding y copias de estados. No permiten atribuir los tiempos de Windows a una función concreta: hace falta el perfil CUDA de esta nueva entrega. No se aumentó el reloj ni se recortó la búsqueda para declarar superada la prueba.

## Cambios

- Preparación de buffers en NumPy y conversión final a tensores CPU. Los valores del esquema v7 se contrastan bit a bit contra el encoder anterior sobre las cinco trayectorias completas; no se eliminan señales.
- Caché acotada a 4,096 entradas para normalizar identificadores. No guarda estados de combate ni predicciones; no puede mezclar experiencias o pesos.
- No se cambia el árbol de búsqueda, su profundidad, los pesos ni la precisión de inferencia.
- `official_benchmark --device cpu|cuda|auto` permite elegir y registrar el dispositivo. Pedir CUDA sin tenerla produce un error. Cada timeout registra el turno, lado, fase y tiempo transcurrido; los percentiles se identifican como tiempos de decisiones exitosas.
- Los resúmenes muestran partidas completadas y timeouts por separado y eliminan texto histórico que no correspondía a corridas nuevas. `compare_checkpoints` compara candidato contra original con bloques emparejados y rechaza configuraciones distintas.

## Validación local completada

Pasaron 585 pruebas, incluida igualdad bit a bit del encoder en 297 estados de cinco trayectorias. Ocho partidas con ambos checkpoints congelados terminaron en CPU/macOS a profundidad 2 y 30 s sin timeouts, acciones inválidas ni fallbacks; 702 solicitudes no mostraron discrepancias de menús legales/PP. Las sondas originales tardaron 16.26 s y 4.79 s. Esta comprobación no establece mejora de juego ni valida la RTX. [Evidencia y hashes](alignment/performance-results.json).

## Ejecutar en la máquina de Santiago

Desde la raíz del clon del fork, con `.venv` activado. Conservar los directorios originales de entrenamiento/evaluación. Los comandos siguientes suponen que siguen en `runs/` como en la entrega anterior.

```sh
git pull --ff-only origin codex/simulator-training-corrections
npm --prefix tools/showdown ci
python -m pip check
python -c "import torch; assert torch.cuda.is_available(), 'Revisar CUDA antes de continuar'; print(torch.__version__,torch.cuda.get_device_name(0))"
python -c "from pathlib import Path; import hashlib; p=Path('runs/pilot-gpu-v7-60s-01/glaubermon_rebel_latest.pt'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='3a4717efe548bca8514c649dd63175ed500f93ec7bdd9d4e053471b23ed10e0e'; print('Candidato entregado OK')"
```

Pruebas en PowerShell (restaurar CUDA después):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m pytest tests/ -q -rs
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

En Bash:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q -rs
```

Ejecutar el proceso completo, igual en Bash y PowerShell:

```sh
python -u -m glaubermon.evaluation.performance_recovery --candidate runs/pilot-gpu-v7-60s-01/glaubermon_rebel_latest.pt --original-trace runs/eval-original-v7-01/hybrid-000.jsonl --candidate-trace runs/eval-candidate-v7-01/hybrid-000.jsonl --showdown tools/showdown/node_modules/pokemon-showdown --output runs/recovery-v7-01 --games 100 --devices cpu cuda --decision-seconds 30
```

Ese comando:

1. Reproduce la última decisión pendiente de las trazas 000 con cada checkpoint en CPU y CUDA, sin ejecutar nuevas partidas ni entrenar. Registra tiempos, llamadas a la red, estrategia, acción y hashes. Las mediciones no activan cProfile para evitar su sobrecoste.
2. Elige el dispositivo con menor tiempo máximo en esas dos muestras. Esto es una selección provisional que debe superar partidas completas; no una afirmación de que ese dispositivo sea siempre superior. Los errores de las sondas se conservan en `recovery.json`.
3. Ejecuta dos partidas por modo y checkpoint en el dispositivo elegido, con el límite original de 30 s y un proceso. Si aparece un timeout, error o partida incompleta, se detiene; no aumenta el límite ni poda la búsqueda automáticamente.
4. Si la prueba anterior pasa, ejecuta cien partidas por modo con cada checkpoint (400 partidas en esa etapa), mismas semillas/equipos/dispositivo, y produce la comparación emparejada. Si hay fallos, conserva evidencia y se detiene para diagnóstico. Dos partidas son comprobación operativa, no evidencia de mejora.

La carpeta de salida debe ser nueva. Para una comprobación corta antes de la etapa larga, usar `--games 2` y otro `--output`; aun así se comparan CPU/CUDA y ambos modelos. Si se pide solo `--devices cpu`, no habrá medición CUDA ni se afirmará que se probó la RTX.

## Diagnóstico más detallado de un frame

Para obtener el perfil de funciones de una traza sin responder:

```sh
python -m glaubermon.evaluation.replay_decision --trace runs/eval-candidate-v7-01/hybrid-000.jsonl --checkpoint runs/pilot-gpu-v7-60s-01/glaubermon_rebel_latest.pt --side 1 --depth 2 --device cuda --profile --output runs/profile-candidate-cuda-01.json
```

El JSON identifica el estado por hash de la traza y conserva la estrategia y acción. Los archivos `.pstats` y `.profile.txt` detallan funciones; cProfile añade sobrecoste. Para comparar tiempos de dispositivos usar las sondas sin `--profile`. La reconstrucción entrega únicamente el canal del lado indicado y restaura su historial de elecciones. Para `hybrid-001.jsonl` el bot estaba en el lado 2: usar `--side 2`. La herramienta exige una traza que termine en una decisión pendiente.

## Entrega de resultados

Compartir `runs/recovery-v7-01/` completa. Contiene `recovery.json`, sondas, logs, manifiestos, partidas, resúmenes y `comparison.json` si la evaluación termina. No sobrescribir la entrega previa ni publicar trazas privadas en GitHub.

```sh
python -c "import shutil; shutil.make_archive('entrega-recovery-v7-01','zip',root_dir='runs',base_dir='recovery-v7-01')"
```

La validación local es en CPU/macOS. El resultado del proceso en Windows/RTX 4090 y la comparación completa del candidato siguen pendientes hasta recibir esa nueva evidencia. Los cambios del encoder preservan sus números, pero cambian su hash: no reanudar el experimento de entrenamiento antiguo con esta versión; este procedimiento solo evalúa pesos congelados.
