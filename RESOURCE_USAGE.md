# 🔍 Análisis de Uso de Recursos - M4 Pro

**Fecha:** 2026-03-28
**Process ID:** 13229
**Command:** `python train.py --config configs/disaster.yaml --models siglip_sigmoid --datasets crisisMMD`

---

## 🖥️ Hardware Disponible

```bash
Model Name: MacBook Pro
Chip: Apple M4 Pro
CPU Cores: 12 performance (P) + 4 efficiency (E) = 16 total
GPU: 20-core GPU (MPS backend)
Memory: 24 GB unified
```

---

## 📊 Uso Actual del Training

### CPU Usage

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **CPU Total** | **8.9%** | Solo ~1 de 12 cores usados |
| **Threads activos** | 11 | num_workers=4 + main thread |
| **Estado** | **U+ (Uninterruptible)** | ⚠️ I/O bottleneck |

**Problema:** Solo usa 1 core efectivamente. Los otros 11 cores están **idle**.

---

### Memory Usage

| Métrica | Valor | Estado |
|---------|-------|--------|
| **Process RSS** | ~126 MB | ✅ Bajo (solo el proceso) |
| **Compressed Memory** | **2.06 GB** | ⚠️ Alto (memory pressure) |
| **Total System Used** | 23 GB / 24 GB | ⚠️ Casi al límite |

**Problema:** 2GB comprimidos indica que el sistema está haciendo swap de memoria a disco.

---

### GPU (MPS)

| Métrica | Valor | Estado |
|---------|-------|--------|
| **MPS Available** | True | ✅ Detectado |
| **MPS Built** | True | ✅ Compilado con soporte |
| **GPU Usage** | ??? | ❓ No se puede medir sin tools especiales |

**Incierto:** MPS está activo pero no podemos ver % real de GPU usage.

---

## 🔍 Diagnóstico del Problema

### Estado del Proceso: `U+` (Uninterruptible Sleep)

**Qué significa:**
- El proceso está bloqueado esperando **I/O** (disk/network)
- No está usando CPU porque está esperando que el disco le entregue datos
- **NO es un deadlock**, es un bottleneck de carga de datos

**Por qué pasa:**
1. `num_workers=4` crea 4 procesos para cargar imágenes
2. Si el disco es lento (HDD) o las imágenes están comprimidas, los workers se bloquean
3. El GPU está idle esperando que lleguen los datos

---

## 📈 Optimizaciones Recomendadas

### ✅ YA CONFIGURADO (Bien)

```yaml
# configs/disaster.yaml
batch_size: 32          # ✅ Apropiado para MPS
num_workers: 4          # ✅ Correcto para MPS
device: "mps"           # ✅ GPU acceleration activada
```

### ⚠️ PROBLEMAS DETECTADOS

| Problema | Severidad | Solución |
|----------|-----------|----------|
| **I/O bottleneck** | 🔴 ALTA | Aumentar `num_workers` o mover dataset a SSD |
| **Bajo uso de CPU** | 🟡 MEDIA | `num_workers=8` para saturar más cores |
| **Memory pressure** | 🟡 MEDIA | Reducir `batch_size` a 16 o cerrar otras apps |
| **Sin métricas de GPU** | 🟢 BAJA | Instalar `asitop` para monitorear MPS real |

---

## 🚀 Cambios Recomendados

### Opción 1: Aumentar num_workers (Primer intento)

```yaml
# configs/disaster.yaml
num_workers: 8  # De 4 → 8
```

**Por qué:** Más workers = más carga de imágenes en paralelo = menos I/O wait

**Riesgo:** Usa más RAM (pero tenes 24GB, deberías estar bien)

---

### Opción 2: Reducir batch_size si hay OOM

```yaml
# Si te sale "Out of Memory"
batch_size: 16  # De 32 → 16
```

**Por qué:** Menos memoria por batch = menos pressure

---

### Opción 3: Mover dataset a SSD

```bash
# Si tus imágenes están en HDD externo
cp -r dataset/damage_dataset ~/SSD/dataset/damage_dataset
```

**Por qué:** SSD es 10-50x más rápido que HDD para I/O aleatorio

---

### Opción 4: Instalar monitor de MPS

```bash
pip install asitop
sudo asitop
```

**Por qué:** Para ver el uso real de GPU/MPS en tiempo real

---

## 📋 Plan de Acción

1. **AHORA (sin reiniciar):**
   - Observar si el training avanza (aunque lento)
   - Monitorear con `top -pid 13229` cada 30 segundos

2. **DESPUÉS (siguiente training):**
   - Aumentar `num_workers: 8` en el YAML
   - Si hay OOM, bajar `batch_size: 16`

3. **FUTURO:**
   - Mover dataset a SSD
   - Instalar `asitop` para monitorear MPS

---

## 🎯 Conclusión

**¿Estamos aprovechando el M4 Pro?**

| Recurso | Uso Actual | Potencial | Aprovechamiento |
|---------|------------|-----------|-----------------|
| CPU (12 cores) | ~1 core (8.9%) | 12 cores | **8%** ⚠️ |
| GPU (MPS) | ??? | 20 cores | **???** ❓ |
| RAM (24GB) | 23 GB | 24 GB | **96%** ⚠️ |

**Veredicto:** ⚠️ **SUB-UTILIZADO**

- **CPU:** Solo usando 1 core por I/O bottleneck
- **Memory:** Casi al límite (otras apps + training)
- **GPU:** Probablemente idle esperando datos

**Acción inmediata:** Aumentar `num_workers` de 4 a 8 para reducir I/O wait.

---

**Generado:** 2026-03-28
