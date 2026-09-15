# Optimización de búsqueda sin eliminar alternativas

La corrección reúne hojas de matrices de reemplazo independientes antes de evaluar la red. Conserva cada combinación de cambios, los ataques pendientes, el orden temporal del juego y la resolución de matrices hijas antes de sus padres. Las hojas se procesan en lotes de hasta 128; se liberan sus referencias después de cada lote. El límite controla memoria, no profundidad ni número de opciones.

El encoder directo elimina los tensores intermedios por movimiento y por condición. Las características v7 siguen siendo iguales bit a bit y los tensores públicos son independientes. Los checkpoints permanecen congelados.

Cambiar la forma de los lotes puede cambiar los últimos decimales de punto flotante; no implica cambiar pesos o usar menor precisión. Por eso se contrastan valores, distribuciones y acciones, además de repetir partidas. Este cambio no certifica una mejora de winrate ni garantiza por sí mismo el límite en otra máquina.

La copia especializada de efectos del experimento anterior fue descartada y no forma parte de esta versión. Tampoco se añadió poda heurística de alternativas.

## Comparaciones locales de los frames

En los frames originales, las invocaciones de la red bajaron de 3,312 a 226 y de 1,905 a 269. En la sonda del candidato bajaron de 437 a 107. Las acciones elegidas se conservaron; la mayor diferencia de valor observada en estas comparaciones fue aproximadamente 4.9e-9 y la de probabilidad 2.5e-11. Estas comparaciones verifican casos concretos, no todas las posiciones posibles.

## Entrega única

`performance_recovery` mide ambos checkpoints, verifica partidas completas y después compara los resultados. Si la comprobación corta termina por timeout, ahora ejecuta automáticamente `diagnose_timeouts` sobre ese checkpoint y conserva el perfil dentro de la misma carpeta. Un error al generar el perfil no convierte la partida fallida en un éxito. La carpeta de salida debe ser nueva y se entrega completa.

Los ocho archivos originales de checkpoints y metadatos se verificaron byte a byte contra el respaldo anterior. Esta optimización no ejecuta actualizaciones de entrenamiento.

## Validación de la entrega

595 pruebas aprobadas y ocho partidas completas en CPU/macOS, con ambos checkpoints, profundidad 2 y 30 s por decisión. Sin timeouts, acciones inválidas ni fallbacks. El auditor verificó 702 solicitudes sin discrepancias de menús legales/PP. Máxima latencia observada por checkpoint: 9.61 s original y 6.67 s candidato. Esta tanda es una comprobación operativa, no evidencia de mejora de winrate. [Resultados y hashes](alignment/pooled-search-results.json).
