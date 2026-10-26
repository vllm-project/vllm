# Machine Specs - myia-ai-01

Configuration matérielle et logicielle de la machine de production vLLM.

**Dernière mise à jour** : 2026-10-26 (topologie GPU remise à jour ; le reste du document date de 2026-02-17)

---

## Hardware

### CPU
- **Modèle** : Intel Core i9-14900KF
- **Cores** : 24 cores physiques
- **Threads** : 32 threads logiques (Hyper-Threading)
- **Architecture** : Raptor Lake (14ème génération)
- **TDP** : 125W base, 253W turbo

### RAM
- **Capacité totale** : **192 GB** DDR5
- **Type** : DDR5 (probablement 4800-5600 MHz)
- **Configuration** : 4x 48GB ou 6x 32GB (à confirmer)

### GPU
- **3x NVIDIA GeForce RTX 4090**
  - VRAM : 24 GB GDDR6X par GPU (72 GB total)
  - Architecture : Ada Lovelace (SM89)
  - CUDA Cores : 16,384 par GPU
  - Tensor Cores : 512 (4ème génération)
  - Driver version : **591.74**
  - Bus : PCIe 4.0 x16
  - TDP : 450W par GPU

- **Topology** :
  - GPUs 0 et 1 : Bus PCIe plus rapide → **Swift-1.5-Qwen3.8-27B en TP=2** (prod depuis 2026-09-25 ; auparavant GLM-4.7-Flash puis Qwen3.5/3.6-35B-A3B)
  - GPU 2 : Bus PCIe plus lent → **libéré le 2026-05-01** pour les entraînements CoursIA (plus de modèle résident ; auparavant OmniCoder-9B puis Kokoro TTS, tous deux archivés/migrés)
  - **Pas de NVLink** (limitation RTX 4090)

### Stockage
- **Disque D:** (principal)
  - Capacité : **932 GB** (probablement 1TB NVMe SSD)
  - Utilisé : 73 GB
  - Disponible : 860 GB
  - Filesystem : NTFS (Windows)

### Alimentation
- **Estimation** : 2000W+ PSU
  - 3x RTX 4090 = 1350W pic
  - i9-14900KF = 253W turbo
  - Reste système + marge

---

## Software Stack

### Système d'Exploitation
- **OS hôte** : Windows 11 Pro 10.0.26200
- **Subsystem** : WSL2 (Windows Subsystem for Linux)
- **Shell** : bash (Unix shell syntax)

### Container Runtime
- **Docker Desktop** pour Windows
- **Version** : (à vérifier avec `docker --version`)
- **NVIDIA Container Toolkit** : Activé
- **Volumes persistants** :
  - `vllm-compile-cache` : Cache torch.compile (~5-10 GB)
  - Bind mounts : `d:/vllm/models` (modèles locaux)

### NVIDIA Stack
- **Driver** : 591.74
- **CUDA** : 12.4+ (inclus dans images Docker vLLM)
- **cuDNN** : Inclus dans images vLLM

---

## Services Hébergés

### 🤖 vLLM Inference Servers

#### GLM-4.7-Flash (Production)
- **Container** : `myia_vllm-medium-glm`
- **Image** : `profiles-vllm-medium-glm` (custom Dockerfile)
- **Port** : 5002
- **GPUs** : 0, 1 (Tensor Parallelism TP=2)
- **Modèle** : QuantTrio/GLM-4.7-Flash-AWQ (31B MoE, 3B actifs)
- **Quantization** : AWQ 4-bit (~19GB)
- **Context** : 128K tokens
- **Performance** :
  - Decode : 44.6 tok/s (single user)
  - Concurrent : 183.8 tok/s (5 users)
  - Tool latency : 1.9s
- **Optimizations** :
  - CUDA graphs + torch.compile
  - Inductor autotune (+30% concurrent)
  - FlashInfer MoE (+13% throughput)
  - Expert Parallelism
  - Prefix caching
- **Status** : Up 5 minutes (healthy)
- **Config** : `myia_vllm/configs/docker/profiles/medium-glm.yml`

#### ZwZ-8B Vision (Mini)
- **Container** : `myia_vllm-mini-zwz`
- **Image** : `vllm/vllm-openai:nightly`
- **Port** : 5001
- **GPU** : 2 (standalone)
- **Modèle** : ZwZ-8B-AWQ-4bit (vision model)
- **Quantization** : AWQ 4-bit (~9GB, vision encoder BF16)
- **Context** : 128K tokens
- **Tool calling** : Oui (hermes parser)
- **Reasoning** : Non (pas de thinking mode)
- **Status** : Up 7 hours (healthy)
- **Config** : `myia_vllm/configs/docker/profiles/mini-zwz.yml`

---

### 🌐 Open-WebUI Instances

#### Instance Principale (MYIA)
- **Container** : `myia-open-webui-open-webui-1`
- **Image** : `ghcr.io/open-webui/open-webui:cuda`
- **Port** : 2090
- **GPU** : Partagé (pour embedding/reranking)
- **Status** : Up 21 hours (healthy)
- **Services associés** :
  - **Kokoro TTS** : `myia-open-webui-kokoro-tts-1` (port 8880)
  - **Whisper STT** : `myia-open-webui-whisper-stt-adapter-1` (port 8787)
  - **Pipelines** : `myia-open-webui-pipelines-1`
  - **Redis** : `myia-open-webui-redis-1` (port 6351)
  - **Tika** : `myia-open-webui-tika-1` (port 9917)

#### Instance EPF GenAI
- **Container** : `epf-genai-open-webui-open-webui-1`
- **Port** : 3013
- **Status** : Up 44 hours (healthy)
- **Redis** : port 6371
- **Tika** : port 9919

#### Instance ECE
- **Container** : `ece-open-webui-open-webui-1`
- **Port** : 3012
- **Status** : Up 44 hours (healthy)
- **Redis** : port 6369
- **Tika** : port 9920

#### Instance EPF
- **Container** : `epf-open-webui-open-webui-1`
- **Port** : 3010
- **Status** : Up 44 hours (healthy)

---

### 🗄️ Databases

#### PostgreSQL (Open-WebUI)
- **Container** : `open-webui-postgres`
- **Image** : `postgres:16-alpine`
- **Port** : 5432
- **Status** : Up 44 hours (healthy)
- **Usage** : Base de données principale pour Open-WebUI

#### MariaDB (WordPress LivresAgites)
- **Container** : `livresagites-db-1`
- **Image** : `mariadb:10.11`
- **Port** : 3306 (interne)
- **Status** : Up 44 hours (healthy)
- **PhpMyAdmin** : `livresagites-phpmyadmin-1` (port 8081)

---

### 🔍 Vector Databases (Qdrant)

#### Qdrant Production
- **Container** : `qdrant_production`
- **Image** : `qdrant/qdrant:latest`
- **Ports** : 6333 (HTTP), 6334 (gRPC)
- **Status** : Up 44 hours
- **Usage** : Vector database pour TOUS les services (Open-WebUI, MCP agents, RooSync semantic search, etc.)

#### Qdrant Students
- **Container** : `qdrant_students`
- **Image** : `qdrant/qdrant:latest`
- **Ports** : 6335 (HTTP), 6336 (gRPC)
- **Status** : Up 44 hours
- **Usage** : Espace isolé pour étudiants/testing/expérimentation

---

### 🔧 Utility Services

#### SearXNG (Meta Search Engine)
- **Container** : `searxng`
- **Image** : `searxng/searxng:latest`
- **Port** : 8181
- **Status** : Up 44 hours
- **Usage** : Search backend pour agents LLM

#### Tika (Document Parsing)
- **Container** : `tika`
- **Image** : `apache/tika:latest-full`
- **Port** : 9918
- **Status** : Up 44 hours
- **Usage** : Extraction de texte depuis PDF/DOCX/etc.

#### Redis (Standalone)
- **Container** : `redis`
- **Image** : `valkey/valkey:8-alpine`
- **Port** : 6379 (interne)
- **Status** : Up 44 hours
- **Usage** : Cache général

---

### 📝 WordPress (LivresAgites)

#### WordPress
- **Container** : `livresagites-wordpress-1`
- **Image** : `wordpress:6.8.3-php8.2-apache`
- **Port** : 8092
- **Status** : Up 44 hours (healthy)

#### WordPress CLI
- **Container** : `livresagites-wordpress_cli-1`
- **Image** : `wordpress:cli-2.9.0-php8.2`
- **Status** : Up 44 hours
- **Usage** : Maintenance WordPress en ligne de commande

---

## Utilisation GPU

### Répartition actuelle (nvidia-smi)

| GPU | Service | VRAM Utilisée | Utilisation | Temp | Power |
|-----|---------|---------------|-------------|------|-------|
| **GPU 0** | GLM-4.7-Flash (TP rank 0) + Windows UI | **24.0 GB / 24.5 GB** | **98%** | 29°C | 17W |
| **GPU 1** | GLM-4.7-Flash (TP rank 1) | **23.7 GB / 24.5 GB** | **97%** | 29°C | 18W |
| **GPU 2** | ZwZ-8B Vision + Windows UI | **20.8 GB / 24.5 GB** | **85%** | 27°C | 26W |

**Total utilisé** : **68.5 GB / 72 GB (95% d'utilisation)**
**Marge disponible** : ~3.5 GB (principalement Windows UI overhead sur GPU 0 et 2)

### Notes
- **GPUs quasi saturés** : Peu de marge pour services additionnels sur GPUs actuels
- **Windows UI overhead** : ~500 MB par GPU pour explorer.exe, Chrome, VS Code, etc.
- **GLM KV cache** : Bénéficie de MLA (54 KB/token) → permet 128K context malgré saturation VRAM
- **Capacité résiduelle limitée** : Pour ajouter des services, considérer upgrade vers 4x GPUs ou RTX avec plus de VRAM

---

## Réseau & Ports

### Ports Exposés (localhost)

| Port | Service | Accès |
|------|---------|-------|
| 2090 | Open-WebUI Principal | Public |
| 3010 | Open-WebUI EPF | Public |
| 3012 | Open-WebUI ECE | Public |
| 3013 | Open-WebUI EPF GenAI | Public |
| 5001 | vLLM ZwZ-8B | API interne |
| 5002 | vLLM GLM-4.7-Flash | API interne |
| 5432 | PostgreSQL | Interne |
| 6333-6334 | Qdrant Production | Interne |
| 6335-6336 | Qdrant Students | Interne |
| 6351 | Redis MYIA | Interne |
| 6369 | Redis ECE | Interne |
| 6371 | Redis EPF GenAI | Interne |
| 8081 | PhpMyAdmin | Admin |
| 8092 | WordPress LivresAgites | Public |
| 8181 | SearXNG | Interne |
| 8880 | Kokoro TTS | Interne |
| 9917-9920 | Tika (4 instances) | Interne |

### Reverse Proxy
- **Probablement** : Nginx ou Traefik devant les services publics
- **Domaines** : `*.text-generation-webui.myia.io` (à confirmer)

---

## Performance & Monitoring

### Métriques vLLM
- **Prometheus** : Exposé sur `/metrics` de chaque service vLLM
- **Logs** : Middleware JSONL à `/logs/chat_completions.jsonl`
- **Healthcheck** : HTTP GET sur `/health` (intervalle 30s)

### Uptime
- **vLLM services** : Redémarrage récent (5 min - 7h)
- **Open-WebUI + DBs** : 21-44 heures d'uptime continu
- **Restart policy** : `unless-stopped` sur tous les containers critiques

---

## Notes de Déploiement

### vLLM Configuration Critique
1. **Bind mounts Windows** : Utiliser `d:/vllm/models`, PAS `/mnt/d/vllm/models`
2. **GPU memory** : 0.92 max (0.95 cause OOM)
3. **Warmup obligatoire** : Lancer `warmup_glm.py` après startup
4. **Compile cache** : Volume nommé `vllm-compile-cache` persiste entre restarts

### Open-WebUI Configuration
- **CUDA image** : Nécessaire pour GPU acceleration (embedding/reranking)
- **Pipelines** : Permet RAG, web search, custom tools
- **Redis** : Cache essentiel pour performance

### Qdrant Configuration
- **Production vs Students** : Isolation des données
- **Persistence** : Volumes Docker pour durabilité

---

## Backup Strategy

### À Définir
- [ ] Backup automatique PostgreSQL
- [ ] Backup automatique Qdrant vectors
- [ ] Backup configs Docker Compose
- [ ] Backup middleware logs
- [ ] Snapshot snapshots modèles quantisés

---

## Évolutions Futures

### Court Terme
- [ ] Investiguer régression vLLM v0.16 (55→44 tok/s)
- [ ] Tester GLM-4.6V-Flash quand vLLM supporte Glm4vForConditionalGeneration
- [ ] Optimiser utilisation GPU 2 (38% seulement)

### Moyen Terme
- [ ] Ajouter monitoring Grafana + Prometheus
- [ ] Tester modèles FP8 (GLM-4.7-Flash-FP8 pour MTP speculative decoding)
- [ ] Évaluer upgrade RAM vers 256 GB (si nécessaire)

### Long Terme
- [ ] Upgrade vers RTX 5090 (si gain significatif VRAM/perf)
- [ ] Investiguer multi-node avec Infinity Fabric (si AMD MI300X viable)

---

**Machine ID** : `myia-ai-01`
**Propriétaire** : MYIA
**Usage** : Production LLM inference + Open-WebUI multi-instances
**Localisation** : (à préciser)
