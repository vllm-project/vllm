#!/bin/bash

# ============================================================
# AUTOMATED SETUP SCRIPT FOR LOCAL AI STACK
# ============================================================

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          LOCAL AI STACK - AUTOMATED SETUP                  ║"
echo "║  LibreChat + mem0 + Qdrant + Ollama                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================
# STEP 1: Prerequisites Check
# ============================================================

echo -e "${BLUE}[1/8] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check disk space (need at least 50GB free)
available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$available_space" -lt 50 ]; then
    echo -e "${YELLOW}⚠️  Warning: Low disk space. Need at least 50GB, have ${available_space}GB${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# ============================================================
# STEP 2: Generate Secrets
# ============================================================

echo -e "${BLUE}[2/8] Generating secure secrets...${NC}"

# Check if librechat.env already has secrets
if grep -q "your_jwt_secret_change_this" librechat.env 2>/dev/null; then
    echo "Generating new secrets..."

    # Generate secrets
    JWT_SECRET=$(openssl rand -hex 32)
    JWT_REFRESH_SECRET=$(openssl rand -hex 32)
    CREDS_KEY=$(openssl rand -hex 16)
    CREDS_IV=$(openssl rand -hex 8)
    DB_PASSWORD=$(openssl rand -hex 16)

    # Update .env file
    sed -i.bak "s/your_jwt_secret_change_this/$JWT_SECRET/" librechat.env
    sed -i.bak "s/your_refresh_secret_change_this/$JWT_REFRESH_SECRET/" librechat.env
    sed -i.bak "s/your_32_character_creds_key_here/$CREDS_KEY/" librechat.env
    sed -i.bak "s/your_16_character_creds_iv_here/$CREDS_IV/" librechat.env
    sed -i.bak "s/your_secure_password_here/$DB_PASSWORD/g" librechat.env

    # Update docker-compose.yml
    sed -i.bak "s/your_secure_password_here/$DB_PASSWORD/g" docker-compose.yml

    # Update init script
    sed -i.bak "s/your_secure_password_here/$DB_PASSWORD/g" init-scripts/01-init-databases.sql

    echo -e "${GREEN}✅ Secrets generated and configured${NC}"
else
    echo -e "${YELLOW}ℹ️  Secrets already configured${NC}"
fi

echo ""

# ============================================================
# STEP 3: Create Directory Structure
# ============================================================

echo -e "${BLUE}[3/8] Creating directory structure...${NC}"

mkdir -p mongodb_data redis_data backups ssl

echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# ============================================================
# STEP 4: Start Infrastructure Services
# ============================================================

echo -e "${BLUE}[4/8] Starting infrastructure services...${NC}"

docker compose up -d qdrant postgres mongodb redis ollama

echo "Waiting for services to be ready..."
sleep 15

echo -e "${GREEN}✅ Infrastructure services started${NC}"
echo ""

# ============================================================
# STEP 5: Pull Ollama Models
# ============================================================

echo -e "${BLUE}[5/8] Pulling Ollama models (this may take a while)...${NC}"

./scripts/pull-models.sh

echo -e "${GREEN}✅ Ollama models ready${NC}"
echo ""

# ============================================================
# STEP 6: Initialize Qdrant Collections
# ============================================================

echo -e "${BLUE}[6/8] Initializing Qdrant collections...${NC}"

sleep 5  # Give Qdrant a moment to fully start
./scripts/init-qdrant.sh

echo -e "${GREEN}✅ Qdrant collections initialized${NC}"
echo ""

# ============================================================
# STEP 7: Start Application Services
# ============================================================

echo -e "${BLUE}[7/8] Starting application services...${NC}"

docker compose up -d openmemory rag_api reranker librechat nginx

echo "Waiting for services to be ready..."
sleep 20

echo -e "${GREEN}✅ Application services started${NC}"
echo ""

# ============================================================
# STEP 8: Health Check
# ============================================================

echo -e "${BLUE}[8/8] Running health check...${NC}"
echo ""

./scripts/health-check.sh

echo ""

# ============================================================
# COMPLETION
# ============================================================

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  SETUP COMPLETE! 🎉                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "📍 Access Points:"
echo "  • LibreChat:        http://localhost:3080"
echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
echo "  • Ollama API:       http://localhost:11434"
echo "  • mem0 API:         http://localhost:8080"
echo "  • RAG API:          http://localhost:8000"
echo ""

echo "🚀 Next Steps:"
echo "  1. Open http://localhost:3080 in your browser"
echo "  2. Create an account (first user is admin)"
echo "  3. Enable memory in Settings → Personalization"
echo "  4. Upload documents to test RAG"
echo "  5. Create custom agents in Settings → Agents"
echo ""

echo "📚 Useful Commands:"
echo "  • View logs:       docker compose logs -f [service]"
echo "  • Restart service: docker compose restart [service]"
echo "  • Stop all:        docker compose down"
echo "  • Health check:    ./scripts/health-check.sh"
echo "  • Backup data:     See README.md for backup instructions"
echo ""

echo "📖 Documentation: See README.md for detailed usage guide"
echo ""

echo -e "${YELLOW}⚠️  Important Security Notes:${NC}"
echo "  • This setup is for LOCAL use only"
echo "  • For production, enable HTTPS and authentication"
echo "  • See README.md 'Security Notes' section"
echo ""

echo -e "${GREEN}Happy coding! 🎨🤖${NC}"
