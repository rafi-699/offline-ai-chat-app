from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

a = Analysis(
    ['src/localpilot/main.py'],
    pathex=['.', 'src'],
    binaries=collect_dynamic_libs('llama_cpp'),
    datas=[
        ('src/localpilot', 'localpilot'),
        ('assets', 'assets'),
        ('.env', '.'), 
    ] + collect_data_files('customtkinter'),
    hiddenimports=[
        'localpilot',
        'localpilot.gui.app',
        'localpilot.server.app',
        'localpilot.config',
        'localpilot.llm.manager',
        'localpilot.rag.index',
        'localpilot.rag.ingest',
        'localpilot.attachments.extract',
        'localpilot.utils.paths',
        'localpilot.utils.security',
        'customtkinter',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'httpx',
        'llama_cpp',
        'anyio',
        'anyio._backends._asyncio',
        'starlette',
        'pydantic',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.http.h11_impl',
        'anyio._backends._trio',
        'h11',
        'starlette.routing',
        'starlette.middleware',
        'starlette.middleware.cors',
        'pydantic_settings',
        'pydantic_settings.env_settings',
    ],
    hookspath=['hooks'],
    runtime_hooks=['hooks/hook-asyncio.py'],
    excludes=['torch', 'tensorflow', 'notebook', 'IPython'],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RafiGPT',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='RafiGPT',
)
