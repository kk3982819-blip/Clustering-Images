# Run Commands

## Web App

Run these commands in PowerShell from the project folder:

```powershell
cd "c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" jinja2 python-multipart
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open the app in your browser:

```text
http://127.0.0.1:8000
```

## CLI Clustering

Run the clustering script directly:

```powershell
.\.venv\Scripts\python.exe cluster_images.py --input ./input --output ./output
```

## Use NVIDIA CUDA

This project automatically uses CUDA when PyTorch can see your NVIDIA GPU.
Check it with:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If it prints `False` or `CPU only`, replace the CPU PyTorch wheel with the CUDA wheel:

```powershell
cd "c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images"
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify again:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected result should include:

```text
True
NVIDIA GeForce RTX ...
```

Then restart the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Fix: PyTorch CUDA Hash Mismatch

If CUDA PyTorch install fails with:

```text
ERROR: THESE PACKAGES DO NOT MATCH THE HASHES
```

The large torch wheel likely downloaded incorrectly or pip reused a bad cache entry. Do not bypass the hash check. Clear the cache and retry:

```powershell
cd "c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images"
.\.venv\Scripts\python.exe -m pip cache purge
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --retries 10 --timeout 120 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

When sky replacement starts, the log should say:

```text
Loading inpainting model ... on cuda
```

If CUDA runs out of memory on a 6 GB laptop GPU, lower the crop size before starting the server:

```powershell
$env:PIXELDWELL_SKY_REGEN_CROP_MAX_SIDE="768"
$env:PIXELDWELL_SKY_REGEN_STEPS="18"
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Sky Replacement Mode

The app now uses the safer sky-only composite mode by default. It replaces only detected sky pixels and preserves the balcony, railing, trees, horizon, and window geometry.

```powershell
$env:PIXELDWELL_SKY_BACKEND="composite"
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Only use the slower SDXL inpainting mode if you explicitly want a generative result:

```powershell
$env:PIXELDWELL_SKY_BACKEND="regenerative"
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

To reduce or increase room relighting:

```powershell
$env:PIXELDWELL_SKY_RELIGHT_INTENSITY="0.25"
```

To reduce or increase sky/environment color matching:

```powershell
$env:PIXELDWELL_SKY_ENV_MATCH_INTENSITY="0.65"
```

To reduce or increase warm floor/sunlight projection matching for sunrise/sunset:

```powershell
$env:PIXELDWELL_SKY_GROUND_LIGHT_INTENSITY="0.72"
```

Sky replacement now uses images in the `reference` folder first:

```text
reference\sunrise
reference\sunset
reference\clear blue sky
reference\cloudy
```

Put one or more sky reference images in those folders. If no matching reference image is found, the app falls back to `static\sky_assets`.

## If Install Uses Python 3.14

Your error happened because `pip` used Python 3.14 from the user install:

```text
AppData\Roaming\Python\Python314
```

`basicsr` does not build correctly there. Use Python 3.11 for this ML project.

Check installed Python versions:

```powershell
py -0p
```

If Python 3.11 is not listed, install it first, then recreate the virtual environment:

```powershell
winget install Python.Python.3.11
cd "c:\Users\Mohammed kaif M\OneDrive\Desktop\clustering images"
Remove-Item -Recurse -Force .\.venv
py -3.11 -m venv .venv
```

## Fix: No Module Named Uvicorn

If you see this error:

```text
No module named uvicorn
```

Run this command, then start the app again:

```powershell
.\.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" jinja2 python-multipart
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
