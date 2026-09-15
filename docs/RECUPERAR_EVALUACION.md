# Validar la optimización completa y comparar los checkpoints

Usar esta entrega después de las correcciones del encoder y la agrupación de hojas de búsqueda. Reemplaza los comandos de diagnóstico anteriores. No se reentrena ni se sustituyen checkpoints: primero hay que validar velocidad y después calidad de juego.

## Qué cambió

- El encoder copia características directamente a buffers propios, sin clonar y reconvertir tensores por movimiento o condición. Las características v7 conservan igualdad bit a bit.
- La búsqueda reúne hojas de reemplazos independientes en lotes de hasta 128 para la red, conservando todas las alternativas y la dependencia entre matrices. Libera las hojas evaluadas para limitar la memoria de los lotes.
- Si la comprobación corta falla por timeout, el proceso recoge automáticamente los perfiles de esos turnos dentro de la misma entrega. No hace falta otro comando de diagnóstico.

En los frames locales del original, las llamadas a la red bajaron de 3,312 a 226 y de 1,905 a 269. En el frame del candidato, de 437 a 107. Se conservaron las elecciones; cambiar el tamaño de lote puede modificar los últimos decimales de los valores. Detalles: [OPTIMIZACION_BUSQUEDA.md](OPTIMIZACION_BUSQUEDA.md).

La RTX todavía debe validar esta versión. La recuperación anterior tuvo dos timeouts a 30 s; el diagnóstico sin profiler tardó 44.03/43.24 s. Conservar esos resultados para comparar. No se afirma haber solucionado el reloj de otra máquina mediante pruebas locales.

## Preparación

Desde la raíz del clon de `FelipeJackFox/glaubermon-max`, en la rama `codex/simulator-training-corrections`, con el entorno `.venv` activado y los archivos recibidos en `runs/`:

```sh
git pull --ff-only origin codex/simulator-training-corrections
npm --prefix tools/showdown ci
python -m pip check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible'; print(torch.__version__,torch.cuda.get_device_name(0))"
python -c "from pathlib import Path; import hashlib; p=Path('runs/pilot-gpu-v7-60s-01/glaubermon_rebel_latest.pt'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='3a4717efe548bca8514c649dd63175ed500f93ec7bdd9d4e053471b23ed10e0e'; print('Candidato OK')"
```

Pruebas en PowerShell, restaurando CUDA al terminar:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m pytest tests/ -q -rs
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

O en Bash:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q -rs
```

No continuar si las pruebas fallan o se omiten pruebas oficiales por no encontrar Showdown.

## Un comando para la validación y evaluación

```sh
python -u -m glaubermon.evaluation.performance_recovery --candidate runs/pilot-gpu-v7-60s-01/glaubermon_rebel_latest.pt --original-trace runs/eval-original-v7-01/hybrid-000.jsonl --candidate-trace runs/eval-candidate-v7-01/hybrid-000.jsonl --showdown tools/showdown/node_modules/pokemon-showdown --output runs/recovery-pooled-v7-01 --games 100 --devices cpu cuda --decision-seconds 30
```

La carpeta de salida debe ser nueva. El proceso mide ambos checkpoints en CPU/CUDA, selecciona provisionalmente el dispositivo más rápido en esas sondas y realiza dos partidas por modo y checkpoint: ocho partidas de comprobación. Conserva el límite de 30 s y profundidad 2. Si pasan, ejecuta cien partidas por modo y checkpoint (400 adicionales), con semillas y configuración emparejadas, y genera `comparison.json`.

Si falla la comprobación corta por timeout, se detiene antes de las 400 y añade `diagnosis-original/` o `diagnosis-candidate/` con las reproducciones sin profiler y con profiler. Los perfiles tienen sobrecoste; los tiempos sin profiler son los comparables. Si falla otra etapa, los registros se conservan y no debe relanzarse con la misma carpeta ni aumentar el reloj automáticamente.

## Compartir la entrega

Tanto si termina como si se detiene por un fallo:

```sh
python -c "import shutil; shutil.make_archive('entrega-recovery-pooled-v7-01','zip',root_dir='runs',base_dir='recovery-pooled-v7-01')"
```

Compartir el ZIP completo. No publicar las trazas privadas en GitHub. No reanudar el entrenamiento anterior con esta versión: cambian hashes del contrato aunque el esquema y los pesos se conserven. Las partidas cortas comprueban funcionamiento; la evaluación completa es la que debe sustentar una conclusión sobre calidad de juego.

## Validación de la entrega

595 pruebas aprobadas y ocho partidas completas en CPU/macOS, con ambos checkpoints, profundidad 2 y 30 s por decisión. Sin timeouts, acciones inválidas ni fallbacks. El auditor verificó 702 solicitudes sin discrepancias de menús legales/PP. Máxima latencia observada por checkpoint: 9.61 s original y 6.67 s candidato. Esta tanda es una comprobación operativa, no evidencia de mejora de winrate. [Resultados y hashes](alignment/pooled-search-results.json).
