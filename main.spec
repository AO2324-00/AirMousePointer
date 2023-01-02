# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=['D:\\research\\github\\AirMousePointer'],
    binaries=[],
    datas=[('C:\\Python310\\Lib\\site-packages\\mediapipe\\modules', 'mediapipe\\modules'),],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
a.datas += [('.\\Assets\\favicon.ico', '.\\Assets\\favicon.ico', 'DATA'),
            ('.\\config.txt', '.\\config.txt', 'DATA'),
            ('.\\Assets\\question.png', '.\\Assets\\question.png', 'DATA'),
            ('.\\Assets\\switch.png', '.\\Assets\\switch.png', 'DATA'),
            ('.\\Assets\\calibration.png','.\\Assets\\calibration.png', 'DATA'),
            ('.\\Assets\\setting.png','.\\Assets\\setting.png', 'DATA'),
            ('.\\Assets\\clear.png', '.\\Assets\\clear.png', 'DATA')]
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AirMousePointer 0.0.3α',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='.\\Assets\\favicon.ico'
)
