# AGENTS.md — guía para asistentes de código

## Proyecto
Fork de Kronos (modelo financiero) con Web UI propia (`webui/app.py`, Flask, puerto 7070) y auditoría empírica de acierto de dirección.

## Entorno
- Windows, Python 3.14, CPU-only, venv en `C:\Projects\Kronos\.venv`
- Workdir SIEMPRE `C:\Projects\Kronos` al lanzar Python (torch carga mal desde otros directorios)
- `HF_HUB_OFFLINE=1` evita ~80s de contacto con HF (los modelos usados están cacheados: small, mini, base, Tokenizer-base, Tokenizer-2k)
- Servidor: `python app.py` dentro de `webui/` (Flask debug con auto-reload de templates). Arrancar con `Start-Process` en background; verificar con `Invoke-WebRequest http://localhost:7070/`.

## Comandos clave
```powershell
# sintaxis
& "C:\Projects\Kronos\.venv\Scripts\python.exe" -m py_compile webui\app.py
# JS del template (extraer <script> a fichero temporal y node --check)
node --check <template_extracted.js>

# tests UI headless (Chrome + puppeteer-core en C:\Users\rodri\AppData\Local\Temp\opencode\uitest)
node ui_horizon.js     # selector de horizonte
node ui_signals.js     # señal confluente + panel multi-señal
```

## Convenciones
- Identidad git SOLO con flags: `git -c user.name="kossto82-ops" -c user.email="kossto82-ops@users.noreply.github.com" commit ...`
- Remotos: `origin` (upstream shiyu-coder), `fork` (kossto82-ops). Push a `fork master`.
- Commit en inglés estilo conventional commits; no hacer push salvo petición explícita.
- Determinismo: para evaluaciones usar siempre `T=0` (greedy). `T>0` es aleatorio e irreproducible.
- Cache de evaluación: clave = archivo|size|mtime|lookback|pred_len (webui/evaluation_results/, gitignored).

## Hechos medidos (no re-derivar, costó ~40 min de CPU)
- Dirección modelo (T=0 small): 41-46% pooled; mom60 55-56%; ★ OOS 41%.
- Modelos con fuerte sesgo bajista 2024-26. Ensemble mini+small+base no mejora.
- Ver README sección "Web UI (fork) + auditoría de acierto".

## Tests / verificación
No hay suite de pytest activa. Verificación mínima tras cambios: py_compile + node --check + un headless de smoke (cargar datos → modelo small → predecir T=0 → comprobar 0 pageErrors).
