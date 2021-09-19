# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['UI_mainwindow.py', 'UI_information.py', 'UI_Violation.py', 'recognition_main.py', 'recognition\\clpr_entry.py', 'recognition\\clpr_recognition.py', 'recognition\\clpr_location.py', 'recognition\\clpr_segmentation.py', 'recognition\\Items.py', 'recognition\\Video.py', 'tools\\Display.py', 'tools\\Geometry.py', 'tools\\Similarity.py', 'recognition\\clpr_detect_network.py', 'recognition\\type_classifier.py', 'tools\\Get_line.py'],
             pathex=['C:\\Users\\13769\\Desktop\\PROGRAMS\\python\\TryTryTry'],
             binaries=[],
             datas=[('tools\\simsun.ttc', '.'), ('mats\\svm.dat', 'mats'), ('mats\\svmchinese.dat', 'mats'), ('mats\\clpr_plate_distinguishing.npy', 'mats'), ('mats\\softmax.npy', 'mats')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='UI_mainwindow',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='UI_mainwindow')
