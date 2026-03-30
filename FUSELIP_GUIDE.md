# 🎯 Guía de FuseLIP-MLP Experiments

**Fecha:** 2026-03-30
**Modelos:** FuseLIP con MLP head (backbone congelado)

---

## 📋 ¿Qué es FuseLIP-MLP?

**Arquitectura:**
- **FuseLIP backbone:** Modelo Vision-Language pre-entrenado (CONGELADO)
- **MLP Head:** 3-layer MLP que se entrena encima
- **Estrategia:** Transfer learning rápido sin fine-tuning completo

**Ventajas:**
- ✅ **Muy rápido** (30 epochs vs 10-50 epochs)
- ✅ **Menos memoria** (backbone congelado)
- ✅ **Buenos resultados** (82.65% accuracy en damage_dataset)

---

## 🚀 Modelos FuseLIP Disponibles

### 1. `fuselip_mlp_image` (Solo Imagen)
```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models fuselip_mlp_image --datasets damage_dataset
```
**Usa:** Solo embeddings de imagen

**Mejor para:** Clasificación basada principalmente en lo que se ve en la foto

---

### 2. `fuselip_mlp_text` (Solo Texto)
```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models fuselip_mlp_text --datasets damage_dataset
```
**Usa:** Solo embeddings de texto (descripción post_text)

**Mejor para:** Análisis de texto acompañante de la imagen

---

### 3. `fuselip_mlp_multimodal` (Imagen + Texto)
```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models fuselip_mlp_multimodal --datasets damage_dataset
```
**Usa:** Concatenación de embeddings de imagen Y texto

**Mejor para:** Aprovechar AMBOS modalidades (recomendado)

---

## 📊 Comparación de Modos

| Modo | Embeddings Usados | Complejidad | Velocidad | Uso Recomendado |
|------|-------------------|-------------|----------|------------------|
| **image** | Solo imagen | Baja | ⚡⚡⚡ Más rápido | Clasificación visual pura |
| **text** | Solo texto | Baja | ⚡⚡⚡ Más rápido | Análisis de captions |
| **multimodal** | Imagen + Texto | Media | ⚡⚡ Rápido | **Lo mejor de ambos mundos** ✅ |

---

## 🎯 Ejemplos de Ejecución

### Eval-only (Probar sin entrenar)
```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models fuselip_mlp_multimodal --datasets damage_dataset --eval-only
```

### Training completo (con caffeine)
```bash
source venv/bin/activate
caffeinate -d python train.py --config configs/disaster.yaml --models fuselip_mlp_multimodal --datasets damage_dataset
```

### Solo crisisMMD dataset
```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models fuselip_mlp_multimodal --datasets crisisMMD
```

### Todos los modos FuseLIP en ambos datasets
```bash
source venv/bin/activate
caffeinate -d python train.py --config configs/disaster.yaml --models fuselip_mlp_image fuselip_mlp_text fuselip_mlp_multimodal
```

---

## ⚙️ Configuración en YAML

Los hiperparámetros de FuseLIP están en `configs/disaster.yaml`:

```yaml
- name: "fuselip_mlp_image"
  checkpoint: "checkpoints/fuselip_mlp_image"
  hyperparams:
    mode: "image"
    embed_dim: 384
    lr: 1.0e-3
    epochs: 30
    weight_decay: 1.0e-4
    batch_size: 64

- name: "fuselip_mlp_text"
  checkpoint: "checkpoints/fuselip_mlp_text"
  hyperparams:
    mode: "text"
    embed_dim: 384
    lr: 1.0e-3
    epochs: 30
    weight_decay: 1.0e-4
    batch_size: 64

- name: "fuselip_mlp_multimodal"
  checkpoint: "checkpoints/fuselip_mlp_multimodal"
  hyperparams:
    mode: "multimodal"
    embed_dim: 384
    lr: 1.0e-3
    epochs: 30
    weight_decay: 1.0e-4
    batch_size: 64
```

---

## 📈 Resultados Esperados

Basado en experimentos previos (`fuselip_mlp_image` en `damage_dataset`):

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 82.65% |
| **Precision** | 83.35% |
| **Recall** | 82.65% |
| **F1 Score** | 82.84% |

**Mejor clase:** `human_damage` (94% precision)
**Peor clase:** `fires` (55% F1)

---

## 🎁 Recomendaciones

### Para empezar rápido:
```bash
source venv/bin/activate
python train.py --config configs/disaster.yaml --models fuselip_mlp_multimodal --datasets damage_dataset
```

### Para comparar los 3 modos:
```bash
source venv/bin/activate
caffeinate -d python train.py --config configs/disaster.yaml --models fuselip_mlp_image fuselip_mlp_text fuselip_mlp_multimodal --datasets damage_dataset
```

### Para crisisMMD (binario):
```bash
source venv/bin/activate
caffeinate -d python train.py --config configs/disaster.yaml --models fuselip_mlp_multimodal --datasets crisisMMD
```

---

## 🔍 Ver Resultados

### Métricas:
```bash
cat outputs/disasternet/damage_dataset/fuselip_mlp_multimodal_metrics.txt
```

### Confusion Matrix:
```bash
open outputs/disasternet/damage_dataset/fuselip_mlp_multimodal_confusion_matrix.png
```

### Checkpoint:
```bash
ls -lh checkpoints/fuselip_mlp_multimodal/
```

---

## 💡 Tips

1. **Start multimodal** - Es el que mejor performance da
2. **Usa caffeine** - Para que el Mac no se duerma
3. **Monitorea logs** - `tail -f outputs/disasternet/run_*.log`
4. **Batch size 64** - FuseLIP usa batch más grande que otros (más rápido)

---

**¡Listo para experimentar con FuseLIP!** 🚀
