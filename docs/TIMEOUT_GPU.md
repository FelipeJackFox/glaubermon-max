# Retomar el piloto tras el timeout de la RTX 4090

Decisión: conservar profundidad 1 y la búsqueda completa de reemplazos; ejecutar con `--decision-seconds 60`. La API y el CLI mantienen 30 segundos por defecto para no cambiar corridas existentes de manera implícita.

El operador reportó 30.35 segundos para el frame del turno 11 de la partida 0: Volt Switch de Iron Treads noquea a Pelipper y genera reemplazos encadenados. Es un reporte del operador; no una medición independiente en esta Mac. Podar estos reemplazos alteraría la política. Aumentar el límite permite completar la misma búsqueda; no constituye una optimización y no garantiza que todos los frames futuros terminen antes de 60 segundos.

El límite se registra en `rebel_meta.json` y se imprime al iniciar. Un timeout conserva la traza, aborta la partida sin agregar sus muestras al buffer e informa número de partida, semilla y límite. El timeout es una condición operativa: no cambia el esquema, las reglas ni el contrato de datos de las búsquedas exitosas. Por eso los checkpoints anteriores siguen siendo compatibles. El hash del entrenador se actualiza en la autorización solo después de las pruebas; no editar manualmente `ALIGNMENT_STATUS.json` en la máquina receptora.

## Actualizar la copia existente

Desde la raíz del clon del fork y con su `.venv` activado. Estos comandos de Git/Python son iguales en Bash y PowerShell:

```sh
git fetch origin
git switch codex/simulator-training-corrections
git pull --ff-only origin codex/simulator-training-corrections
python -m glaubermon.scripts.train_rebel --help
python -c "from pathlib import Path; from glaubermon.evaluation.alignment_gate import require_pilot_alignment; s=require_pilot_alignment('showdown',Path('tools/showdown/node_modules/pokemon-showdown')); print('Alineacion:',s['training_allowed'])"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible'; print(torch.cuda.get_device_name(0))"
```

`--help` debe incluir `--decision-seconds`. No saltar un error del gate ni descartar modificaciones locales para conseguir que pase. Si `origin` apunta al repositorio de Santiago y el PR todavía no está incorporado, usar el fork indicado en [ENTRENAR_GPU.md](ENTRENAR_GPU.md).

Pruebas de regresión en CPU antes de volver a la GPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q -rs
```

En PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m pytest tests/ -q -rs
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Referencia esperada: 575 pruebas aprobadas, ninguna omitida. Volver a verificar CUDA después si se configuró la variable en toda la sesión.

## Repetir el piloto sin sobrescribir la traza fallida

Como falló la partida 0, todavía no hay una actualización del modelo producida por esa partida. Crear otro experimento; conservar `runs/pilot-gpu-v7-01/` para diagnóstico. No copiar su traza fallida al experimento nuevo.

```sh
python -c "from pathlib import Path; Path('runs/pilot-gpu-v7-60s-01').mkdir(parents=True,exist_ok=False)"
git rev-parse HEAD > runs/pilot-gpu-v7-60s-01/source.txt
python -m pip freeze > runs/pilot-gpu-v7-60s-01/environment.txt
nvidia-smi > runs/pilot-gpu-v7-60s-01/gpu.txt
```

Bash:

```bash
set -o pipefail
python -u -m glaubermon.scripts.train_rebel --games 10 --save-every 1 --eval-every 0 --checkpoint-dir runs/pilot-gpu-v7-60s-01 --mechanics-seed 7331 --rollout-backend showdown --showdown-path tools/showdown/node_modules/pokemon-showdown --max-turns 300 --depth 1 --decision-seconds 60 --torch-threads 1 2>&1 | tee runs/pilot-gpu-v7-60s-01/train-001.log
```

PowerShell:

```powershell
python -u -m glaubermon.scripts.train_rebel --games 10 --save-every 1 --eval-every 0 --checkpoint-dir runs/pilot-gpu-v7-60s-01 --mechanics-seed 7331 --rollout-backend showdown --showdown-path tools/showdown/node_modules/pokemon-showdown --max-turns 300 --depth 1 --decision-seconds 60 --torch-threads 1 2>&1 | Tee-Object -FilePath runs/pilot-gpu-v7-60s-01/train-001.log
if ($LASTEXITCODE -ne 0) { throw 'Fallo el piloto: conservar log y traza' }
```

Verificar `Initialized on: cuda`, `Official decision timeout: 60.0 seconds` y, al terminar, los metadatos:

```sh
python -c "import json,torch; from pathlib import Path; p=Path('runs/pilot-gpu-v7-60s-01'); m=json.loads((p/'rebel_meta.json').read_text()); print('Partidas:',m['total_games'],'Dispositivo:',m['device'],'Limite:',m['decision_seconds']); assert m['total_games']==10 and m['device'].startswith('cuda') and m['decision_seconds']==60; w=torch.load(p/'glaubermon_rebel_latest.pt',map_location='cpu',weights_only=True); assert all(torch.isfinite(v).all().item() for v in w.values())"
```

La comprobación de actualización real, continuación y evaluación sigue en [ENTRENAR_GPU.md](ENTRENAR_GPU.md): sustituir **todas** las rutas `runs/pilot-gpu-v7-01` por `runs/pilot-gpu-v7-60s-01`, mantener `--decision-seconds 60` y añadir 90 partidas solo si las diez anteriores completaron sus comprobaciones. `--games` agrega partidas. El benchmark tiene su propio `--decision-seconds`; no cambiarlo entre candidato y original.

Si otro frame supera 60 segundos, detener y compartir la traza correspondiente antes de seguir aumentando el límite. No convertir la partida abortada en empate ni aceptar una acción por defecto. La evaluación en la RTX 4090 continúa pendiente de la repetición del operador.
