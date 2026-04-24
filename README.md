# damageVLMs-experiment

Experimentos de clasificación de desastres con VLMs y baselines ConvNet image-only.

## ConvNet image-only benchmark

Este repo ahora incluye un flujo separado del pipeline VLM para correr baselines ConvNet sobre:

- `damage_dataset`
- `crisisMMD`

Modelos soportados:

- `resnet50`
- `efficientnet_b0`
- `vgg16`

### Entorno local

Usar siempre el entorno virtual local:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-convnets.txt
```

Configurar la ruta local de datasets en `.env`:

```bash
DATASET_ROOT=/Users/emersonherreratolentino/www/laboratorio/dataset
```

### Ejecutar benchmark base

```bash
source .venv/bin/activate
python train.py --config configs/convnets_image_only.yaml
```

### Reporte técnico

El benchmark y su validación multi-seed quedaron documentados en:

- `docs/convnet-image-only-benchmark-report.md`

### Estado actual

- baseline final: `resnet50`
- alternativa operativa: `efficientnet_b0`
- micro-tuning de `efficientnet_b0`: cerrado por ahora
