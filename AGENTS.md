# AGENTS.md

## Objetivo
Este repositorio usa Python 3.10, `uv` como gestor de entorno y dependencias, y `Makefile` como interfaz única para tareas de desarrollo.

## Reglas operativas
- Usa `make sync` para crear o actualizar `.venv`.
- Usa `make lock` cuando cambien dependencias.
- Usa `make format`, `make lint` y `make typecheck` antes de entregar cambios.
- Usa `make check` para validar sin modificar archivos.
- Usa `make test` para ejecutar la suite de `pytest`.
- Valida contra el servidor local en `http://127.0.0.1:8080` con el modelo `unsloth/Mistral-Small-4-119B-2603-GGUF:UD-Q5_K_XL`.
- Para smoke tests del CLI usa `uv run python -m mistral4cli --print-defaults` y `uv run python -m mistral4cli --once "..." --no-stream`.
- Si trabajas con imágenes, usa siempre una imagen de al menos `2x2` píxeles.
- Si trabajas con MCP, usa `mcp.json` en la raíz del repo o `MISTRAL_LOCAL_MCP_CONFIG` para apuntar a otra configuración.
- La REPL debe mantener ayuda clara y accionable: `/help`, `/defaults`, `/tools`, `/reset`, `/system`, `/exit`.
- La UI del CLI debe conservar el estilo retro verde/naranja y un banner ASCII legible en terminales TTY.
- No uses `pip`, `poetry`, `pipenv` ni instalaciones globales para el flujo normal del proyecto.
- Ejecuta código siempre dentro del entorno de `uv` con `uv run ...` o mediante `make ...`.
- Mantén el código compatible con Python 3.10.

## Estilo de desarrollo
- Prefiere `src/` layout para el código de aplicación.
- Mantén funciones pequeñas y con responsabilidad única.
- Tipa las funciones públicas y los límites entre módulos.
- Evita lógica compleja en `__init__.py`; usa módulos explícitos para la entrada CLI.
- Usa `pathlib` para rutas, `dataclasses` para estructuras de datos simples y `logging` para salida operativa.

## Lint y formato
- El formato canónico es `ruff format`.
- La higiene de imports y lint se valida con `ruff check`.
- Si un cambio requiere una excepción de lint, documenta por qué y limita el alcance.
- No desactives reglas de forma global salvo que exista una razón clara y estable.

## Tipado
- `mypy` debe seguir pasando en `src/`.
- Añade anotaciones en funciones nuevas o modificadas.
- Prefiere tipos concretos sobre `Any`.
- Si necesitas un `# type: ignore`, justifícalo y revisa si puede eliminarse después.

## Dependencias
- Añade dependencias solo en `pyproject.toml`.
- Regenera `uv.lock` después de cambiar dependencias.
- Mantén el árbol de dependencias pequeño y justificado.

## Flujo recomendado
1. `make sync`
2. Implementar el cambio
3. `make format`
4. `make lint`
5. `make typecheck`
6. `make test`
7. Si todo pasa, entregar el cambio

## Criterio de calidad
- Acompaña cambios de comportamiento con pruebas cuando aplique.
- Evita romper compatibilidad con Python 3.10.
- Mantén mensajes y APIs simples si el proyecto sigue siendo pequeño.
