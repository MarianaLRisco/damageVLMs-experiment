# 🚀 Comandos para Correr Experiments

**Fecha:** 2026-03-30
**Estado:** Production-ready (100% type-safe, logging activado)

---

## 📋 Requisitos Previos

### 1. Activar entorno virtual
```bash
source venv/bin/activate
```

### 2. Verificar que todo esté listo
```bash
python -c "from pipelines.registry import get_pipeline; print('✅ OK')"
```

---

## 🎯 Comandos Principales

### Eval-Only (Rápido - ~2 min)
**Sin training, solo evaluación zero-shot.** Ideal para probar que todo funciona.

```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --eval-only
```

---

### Un Solo Modelo
**Entrena o evalúa solo el modelo especificado.**

```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models fuselip_mlp_image
```

---

### Un Solo Dataset
**Corre todos los modelos solo en el dataset especificado.**

```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --datasets damage_dataset
```

---

### Modelo + Dataset Específico
**Combinación específica de modelo y dataset.**

```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models siglip_sigmoid --datasets damage_dataset
```

---

### Training Completo (TODOS los modelos y datasets)
**⚠️ Advertencia: Esto tomará MUCHO tiempo (varias horas).**

```bash
source venv/bin/activate
caffeinate -d python train.py --config configs/disaster.yaml
```

**Nota:** `caffeinate -d` evita que el Mac se duerma durante el training largo.

---

## 📁 Modelos Disponibles

### Contrastive (Full Fine-tuning)
- `siglip_sigmoid` - SigLIP base + sigmoid loss
- `siglip_crossentropy` - SigLIP base + cross-entropy loss
- `siglip2_sigmoid` - SigLIP2 + sigmoid loss
- `siglip2_crossentropy` - SigLIP2 + cross-entropy loss

### Two-Stage (LayerNorm + Classifier)
- `siglip_twoStage` - SigLIP two-stage training
- `siglip2_twoStage` - SigLIP2 two-stage training
- `siglip_twoStage_fewshot` - SigLIP two-stage + few-shot (16 samples)
- `siglip2_twoStage_fewshot` - SigLIP2 two-stage + few-shot

### FuseLIP-MLP (Frozen Backbone)
- `fuselip_mlp_image` - Solo imagen
- `fuselip_mlp_text` - Solo texto
- `fuselip_mlp_multimodal` - Imagen + texto

---

## 📊 Datasets Disponibles

### `damage_dataset` (6 clases)
- `non_damage` - Sin daño
- `fires` - Incendios
- `flood` - Inundaciones
- `damaged_infrastructure` - Infraestructura dañada
- `damaged_nature` - Naturaleza dañada
- `human_damage` - Daño humano

### `crisisMMD` (2 clases)
- `informative` - Informativo
- `not_informative` - No informativo

---

## 🔧 Parámetros Adicionales

### Filtrar múltiples modelos
```bash
python train.py --config configs/disaster.yaml --models siglip_sigmoid siglip_twoStage fuselip_mlp_image
```

### Filtrar múltiples datasets
```bash
python train.py --config configs/disaster.yaml --datasets damage_dataset crisisMMD
```

### Cambiar directorio de output
```bash
python train.py --config configs/disaster.yaml --output-dir outputs/test_experiment
```

---

## 📈 Monitoreo de Experiments

### Ver logs en tiempo real
```bash
# Los logs se guardan en outputs/disasternet/
tail -f outputs/disasternet/run_configs_disaster_*.log
```

### Ver checkpoints guardados
```bash
ls -lh checkpoints/
```

### Ver resultados anteriores
```bash
ls -lh outputs/disasternet/*/
cat outputs/disasternet/*/fuselip_mlp_image_metrics.txt
```

---

## 🎯 Ejemplos de Uso

### Ejemplo 1: Probar un modelo nuevo
```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models siglip_sigmoid --datasets damage_dataset --eval-only
```

### Ejemplo 2: Entrenar un modelo específico
```bash
source venv/bin/activate
caffeinate -d python train.py --config configs/disaster.yaml --models fuselip_mlp_image --datasets damage_dataset
```

### Ejemplo 3: Comparar todos los modelos en un dataset
```bash
source venv/bin/activate
caffeinate -d python train.py --config configs/disaster.yaml --datasets damage_dataset
```

---

## ⚠️ Consideraciones de Rendimiento

### En Mac M4 Pro:
- **GPU:** MPS (Metal Performance Shaders) activado por defecto
- **Workers:** 8 (configurado para óptimo rendimiento)
- **Batch size:** 32 (contrastive), 64 (fuselip)

### Si te quedas sin memoria:
Reducir `batch_size` en `configs/disaster.yaml`:
```yaml
hyperparams:
  batch_size: 16  # En lugar de 32 o 64
```

---

## 🐛 Solución de Problemas

### Error: "CUDA not available"
**Solución:** El código detecta automáticamente MPS en Mac. No hacés nada.

### Error: "Out of memory"
**Solución:** Reduce el `batch_size` en el YAML o cierra otras aplicaciones.

### Error: "Dataset not found"
**Solución:** Verificá que `dataset/` exista y tenga la estructura correcta.

---

## 📚 Referencias

- **Configuración:** `configs/disaster.yaml`
- **Logging:** `outputs/disasternet/run_*.log`
- **Métricas:** `outputs/disasternet/*/_metrics.txt`
- **Checkpoints:** `checkpoints/`

---

**Última actualización:** 2026-03-30
