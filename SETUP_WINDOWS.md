# Setup en Windows (GPU NVIDIA o CPU)

## Requisitos previos

- Python 3.11 desde [python.org](https://www.python.org/downloads/) — marcar **"Add Python to PATH"** al instalar
- Git desde [git-scm.com](https://git-scm.com/)

---

## 1. Abrir terminal

Usar **PowerShell** o **Command Prompt** (no Git Bash para activar el venv).

---

## 2. Crear entorno virtual

```powershell
python -m venv venv
```

---

## 3. Activar el entorno

```powershell
venv\Scripts\activate
```

Verás algo así:

```
(venv) C:\Users\tu_usuario\mi_proyecto>
```

> Si sale error de permisos en PowerShell:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 4. Actualizar herramientas básicas

```powershell
pip install --upgrade pip setuptools wheel
```

---

## 5. Instalar PyTorch (primero, según hardware)

**Con GPU NVIDIA (recomendado):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Sin GPU / solo CPU:**
```powershell
pip install torch torchvision torchaudio
```

Verificar que la GPU funciona:
```python
import torch
print(torch.cuda.is_available())   # True si hay GPU NVIDIA
print(torch.cuda.get_device_name(0))
```

---

## 6. Instalar dependencias

```powershell
pip install -r requirements.txt
```

---

## 7. Uso diario

Activar entorno:
```powershell
venv\Scripts\activate
```

Desactivar:
```powershell
deactivate
```

---

## 8. Tips importantes (Windows)

- **No mezclar** pip global con venv
- **No usar** Anaconda si ya usas venv
- `triton` **no va** en requirements.txt — solo funciona en Linux
- Si falla `sentencepiece`:
  ```powershell
  pip install sentencepiece --prefer-binary
  ```
- Si falla `protobuf`:
  ```powershell
  pip install protobuf==3.20.*
  ```

---

## 9. Estructura del proyecto

```
damageVLMs-experiment/
│
├── venv/
├── requirements.txt
├── train.py
├── configs/
├── engine/
├── models/
├── dataset/
└── outputs/
```

---

## Resumen

| | Mac (M4) | Windows (NVIDIA) |
|---|---|---|
| Activar venv | `source venv/bin/activate` | `venv\Scripts\activate` |
| GPU backend | MPS (Apple Silicon) | CUDA |
| PyTorch install | `pip install torch` | `pip install torch --index-url .../cu118` |
| `triton` | No usar | No usar |
| Verificar GPU | `torch.backends.mps.is_available()` | `torch.cuda.is_available()` |
