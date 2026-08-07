cat << 'EOF' > check_versions.py && uv run check_versions.py

import fastapi

import starlette

print(f"FastAPI version: {fastapi.__version__}")

print(f"Starlette version: {starlette.__version__}")

EOF
