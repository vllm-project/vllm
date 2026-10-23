# Guide de Déploiement - Service vLLM Medium (Qwen3-32B-AWQ)

> ⚠️ **DOCUMENT HISTORIQUE (2025-10)** — décrit le déploiement **Qwen3-32B-AWQ**, retiré de production depuis ~novembre 2025 (lignée : GLM-4.7-Flash → Qwen3.5 → Qwen3.6 → Swift-1.5-Qwen3.8-27B, en prod depuis 2026-09-25). Conservé pour ses procédures génériques (pré-validation, monitoring, troubleshooting).
>
> **Pour déployer le service courant** : profile `myia_vllm/configs/docker/profiles/medium-swift15-27b.yml` (image `vllm/vllm-openai:v0.32.0` épinglée, port **5002**, conteneur `myia_vllm-medium-swift15-27b`) — commandes dans `CLAUDE.md` §Common Commands.
>
> **Ne pas suivre tel quel** : `--enable-chunked-prefill` est aujourd'hui **interdit** (force le fallback V0), le prefix caching est **activé** (et accordé `--prefix-match-unit 16`), l'image n'est plus `latest` mais épinglée, et plusieurs chemins cités (`profiles/medium.yml`, `scripts/deploy_medium_monitored.ps1`) **n'existent plus**.

**Dernière mise à jour** : 2025-10-22 (marqué historique le 2026-10-23, passe fossiles siège po-2025)  
**Version** : 1.1-historique

## 📋 Table des Matières
1. [Prérequis Système](#1-prérequis-système)
2. [Configuration Environnement](#2-configuration-environnement)
3. [Validation Pré-Déploiement](#3-validation-pré-déploiement)
4. [Procédure de Déploiement](#4-procédure-de-déploiement)
5. [Monitoring Post-Déploiement](#5-monitoring-post-déploiement)
6. [Troubleshooting Déploiement](#6-troubleshooting-déploiement)
7. [Scripts de Référence](#7-scripts-de-référence)

---

## 1. Prérequis Système

### Matériel Requis

- **GPU** : 2× NVIDIA RTX 4090 (24GB VRAM chacune)
- **RAM** : Minimum 32GB recommandé
- **Disque** : ~50GB disponible pour modèle + cache

### Logiciels Requis

- Docker Desktop avec support GPU (NVIDIA Container Toolkit)
- PowerShell 7+ (pour scripts automation)
- Git (pour gestion version)

### Vérification GPU

```bash
# Vérifier disponibilité 2× RTX 4090
nvidia-smi

# Tester GPU dans conteneur Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Sortie attendue** : Liste des 2 GPUs avec mémoire disponible

---

## 2. Configuration Environnement

### Fichier `.env`

Créer [`myia_vllm/.env`](myia_vllm/.env) avec :

```bash
# API Keys (NE JAMAIS VERSIONNER)
VLLM_API_KEY_MEDIUM=<générer_clé_sécurisée>

# GPU Configuration
CUDA_VISIBLE_DEVICES_MEDIUM=0,1

# Container Configuration
CONTAINER_NAME_MEDIUM=myia_vllm-medium-qwen3
```

**⚠️ SÉCURITÉ** : Fichier `.env` doit être dans [`.gitignore`](myia_vllm/.gitignore)

**📚 Documentation complète** : Voir [`setup/ENV_CONFIGURATION.md`](myia_vllm/docs/setup/ENV_CONFIGURATION.md)

### Configuration Docker Compose

Fichier principal : [`myia_vllm/configs/docker/profiles/medium.yml`](myia_vllm/configs/docker/profiles/medium.yml)

**Paramètres critiques validés (Configuration Optimale - Mission 15)** :

```yaml
services:
  medium-qwen3:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ipc: host
    shm_size: '16gb'
    environment:
      - NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_MEDIUM:-0,1}
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    command:
      - "--model"
      - "Qwen/Qwen3-32B-AWQ"
      - "--gpu-memory-utilization"
      - "0.85"
      - "--enable-chunked-prefill"
      - "--tensor-parallel-size"
      - "2"
```

**Paramètres clés** :
- [`gpu-memory-utilization: 0.85`](myia_vllm/configs/docker/profiles/medium.yml) (vs 0.95 baseline) : Stabilité accrue
- [`enable-chunked-prefill: true`](myia_vllm/configs/docker/profiles/medium.yml) : Lissage utilisation mémoire
- [`enable-prefix-caching: false`](myia_vllm/configs/docker/profiles/medium.yml) : Meilleures performances seul (désactivé)
- [`tensor-parallel-size: 2`](myia_vllm/configs/docker/profiles/medium.yml) : 2× RTX 4090

**📚 Documentation architecture** : Voir [`docker/ARCHITECTURE.md`](myia_vllm/docs/docker/ARCHITECTURE.md)

---

## 3. Validation Pré-Déploiement

### Checklist Sécurité

Source : [`deployment/DEPLOYMENT_SAFETY.md`](myia_vllm/docs/deployment/DEPLOYMENT_SAFETY.md)

- [ ] Fichier `.env` créé et NON versionné
- [ ] Clé API générée (32+ caractères alphanumériques)
- [ ] GPUs détectés (`nvidia-smi` réussit)
- [ ] Docker Desktop lancé et fonctionnel
- [ ] Espace disque suffisant (~50GB libre)
- [ ] Aucun autre conteneur vLLM actif (`docker ps -a`)

### Vérification Automatisée

Utiliser script : [`myia_vllm/scripts/check_ram_usage.ps1`](myia_vllm/scripts/check_ram_usage.ps1)

```powershell
# Vérifier ressources système
pwsh -c ".\myia_vllm\scripts\check_ram_usage.ps1"
```

### Validation Configuration Docker

```bash
# Valider syntaxe YAML
docker compose -f myia_vllm/configs/docker/profiles/medium.yml config

# Vérifier chargement variables .env
docker compose -p myia_vllm --env-file myia_vllm/.env \
  -f myia_vllm/configs/docker/profiles/medium.yml config | grep -i "api_key\|cuda"
```

---

## 4. Procédure de Déploiement

### Méthode 1 : Déploiement Automatisé (RECOMMANDÉ)

**Script** : [`myia_vllm/scripts/deploy_medium_monitored.ps1`](myia_vllm/scripts/deploy_medium_monitored.ps1)

```powershell
# Lancer déploiement avec monitoring intégré
pwsh -c ".\myia_vllm\scripts\deploy_medium_monitored.ps1"
```

**Ce script effectue automatiquement :**
1. Nettoyage containers existants
2. Reconstruction image Docker (`--build --force-recreate`)
3. Lancement service en arrière-plan (`-d`)
4. Monitoring jusqu'à état `(healthy)`
5. Validation health check

**Durée estimée** : 5-10 minutes (chargement modèle + initialisation)

### Méthode 2 : Déploiement Manuel

**Étape 1 : Nettoyage**
```powershell
cd myia_vllm
docker compose -p myia_vllm --env-file .env \
  -f configs/docker/docker-compose.yml \
  -f configs/docker/profiles/medium.yml \
  down --remove-orphans
```

**Étape 2 : Lancement**
```powershell
docker compose -p myia_vllm --env-file .env \
  -f configs/docker/docker-compose.yml \
  -f configs/docker/profiles/medium.yml \
  up -d --build --force-recreate
```

**Étape 3 : Monitoring**
```powershell
# Attendre état (healthy) - Timeout 10 minutes
pwsh -c ".\scripts\monitoring\wait_for_container_healthy.ps1 -Timeout 600"
```

**📚 Script monitoring** : [`scripts/monitoring/wait_for_container_healthy.ps1`](myia_vllm/scripts/monitoring/wait_for_container_healthy.ps1)

---

## 5. Monitoring Post-Déploiement

### Vérification État Container

```powershell
# Vérifier container actif
docker ps --filter "name=myia_vllm-medium-qwen3"

# Exemple output attendu :
# CONTAINER ID   STATUS                    PORTS
# abc123def456   Up 5 minutes (healthy)    0.0.0.0:8000->8000/tcp
```

### Health Check API

```bash
# Test endpoint santé
curl http://localhost:8000/health

# Réponse attendue :
# {"status":"ok"}
```

### Monitoring Continu

**Script** : [`myia_vllm/scripts/monitor_medium.ps1`](myia_vllm/scripts/monitor_medium.ps1)

```powershell
# Monitoring live (logs + ressources)
pwsh -c ".\myia_vllm\scripts\monitor_medium.ps1"
```

**Fonctionnalités** :
- ✅ Surveillance status conteneur toutes les 10s
- ✅ Extraction derniers logs (20 lignes)
- ✅ Détection erreurs critiques (ERROR, FATAL, OOM)
- ✅ Timeout configurable (défaut 10 min)

### Métriques GPU

```bash
# Utilisation GPU en temps réel
watch -n 2 nvidia-smi

# GPU du conteneur spécifique
docker exec myia_vllm-medium-qwen3 nvidia-smi
```

**Utilisation attendue** :
- GPU 0 et 1 : ~90-95% pendant inférence
- Mémoire : ~12 GiB utilisés / 24 GiB par GPU

---

## 6. Troubleshooting Déploiement

### Problème : Container ne démarre pas

**Symptômes :** Container en statut `Exited` ou `Restarting`

**Diagnostic :**
```powershell
# Voir logs d'erreur
docker logs myia_vllm-medium-qwen3
```

**Solutions courantes :**

1. **Erreur API Key** : Vérifier `.env` contient `VLLM_API_KEY_MEDIUM`
   ```powershell
   grep VLLM_API_KEY_MEDIUM myia_vllm/.env
   ```

2. **GPU non disponible** : Vérifier `nvidia-smi` fonctionne
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Mémoire insuffisante** : Réduire `gpu-memory-utilization` à 0.80
   ```yaml
   # Dans medium.yml
   --gpu-memory-utilization 0.80
   ```

### Problème : Health Check échoue

**Symptômes :** Container en statut `(unhealthy)`

**Diagnostic :**
```bash
# Voir détails health check
docker inspect myia_vllm-medium-qwen3 | grep -A 10 Health
```

**Solutions :**
1. Attendre plus longtemps (chargement modèle ~5-10 min)
2. Vérifier logs : `docker logs myia_vllm-medium-qwen3 | tail -50`
3. Chercher "Model loaded successfully" dans les logs

### Problème : API ne répond pas

**Symptômes :** Container `(healthy)` mais `curl` échoue

**Diagnostic :**
```bash
# Tester endpoint principal
curl http://localhost:8000/v1/models
```

**Solutions :**
1. Vérifier port mapping : `docker ps` (doit montrer `0.0.0.0:8000->8000/tcp`)
2. Vérifier firewall n'bloque pas port 8000
3. Voir section dédiée dans [`TROUBLESHOOTING.md`](myia_vllm/docs/TROUBLESHOOTING.md)

### Problème : GPUs Non Détectés

**Symptômes :**
```
RuntimeError: Failed to infer device type
ImportError: libcuda.so.1: cannot open shared object file
```

**Cause Racine** : Configuration Docker Swarm incompatible avec Docker Compose standalone

**Solution** : Utiliser syntaxe Docker Compose correcte dans [`medium.yml`](myia_vllm/configs/docker/profiles/medium.yml)

✅ **CORRECT (Docker Compose standalone)** :
```yaml
runtime: nvidia
ipc: host
shm_size: '16gb'
environment:
  - NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_MEDIUM:-0,1}
```

❌ **INCORRECT (Docker Swarm)** :
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: [gpu]
```

**Validation du Fix :**
```powershell
# Déploiement avec script officiel
pwsh -File scripts/deploy_medium_monitored.ps1

# Le monitoring doit afficher :
# ✅ Automatically detected platform cuda
# ✅ GPU KV cache size: 195,312 tokens
# ✅ Available KV cache memory: 11.92 GiB
# ✅ Status: healthy
```

**📚 Documentation détaillée** : Voir [`deployment/DEPLOYMENT_SAFETY.md`](myia_vllm/docs/deployment/DEPLOYMENT_SAFETY.md)

---

## 7. Scripts de Référence

### Scripts de Déploiement

| Script | Fonction | Usage |
|--------|----------|-------|
| [`deploy_medium_monitored.ps1`](myia_vllm/scripts/deploy_medium_monitored.ps1) | Déploiement automatisé | Recommandé production |
| [`wait_for_container_healthy.ps1`](myia_vllm/scripts/monitoring/wait_for_container_healthy.ps1) | Attente health check | Intégration CI/CD |
| [`monitor_medium.ps1`](myia_vllm/scripts/monitor_medium.ps1) | Monitoring continu | Debugging live |

### Scripts de Validation

| Script | Fonction | Usage |
|--------|----------|-------|
| [`mission15_validation_tests.ps1`](myia_vllm/scripts/testing/mission15_validation_tests.ps1) | Suite tests validation | Post-déploiement |
| [`test_kv_cache_acceleration.ps1`](myia_vllm/scripts/test_kv_cache_acceleration.ps1) | Benchmark KV Cache | Performance |
| [`check_ram_usage.ps1`](myia_vllm/scripts/check_ram_usage.ps1) | Vérification ressources | Pré-déploiement |

### Scripts de Maintenance

Voir [`scripts/maintenance/`](myia_vllm/scripts/maintenance/) et guide dédié [`MAINTENANCE_PROCEDURES.md`](myia_vllm/docs/MAINTENANCE_PROCEDURES.md)

---

## 📚 Documentation Complémentaire

- **Architecture Docker** : [`docker/ARCHITECTURE.md`](myia_vllm/docs/docker/ARCHITECTURE.md)
- **Paramètres Service** : [`docker/MEDIUM_SERVICE_PARAMETERS.md`](myia_vllm/docs/docker/MEDIUM_SERVICE_PARAMETERS.md)
- **Configuration `.env`** : [`setup/ENV_CONFIGURATION.md`](myia_vllm/docs/setup/ENV_CONFIGURATION.md)
- **Optimisation Configuration** : [`OPTIMIZATION_GUIDE.md`](myia_vllm/docs/OPTIMIZATION_GUIDE.md)
- **Résolution Problèmes** : [`TROUBLESHOOTING.md`](myia_vllm/docs/TROUBLESHOOTING.md)
- **Rapport Déploiement 16/10** : [`deployment/DEPLOYMENT_MEDIUM_20251016.md`](myia_vllm/docs/deployment/DEPLOYMENT_MEDIUM_20251016.md)

---

**Fin du Guide de Déploiement vLLM Medium**