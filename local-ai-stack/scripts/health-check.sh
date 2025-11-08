#!/bin/bash

# ============================================================
# HEALTH CHECK SCRIPT FOR ALL SERVICES
# ============================================================

set -e

echo "🏥 Checking health of all services..."
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check HTTP endpoint
check_http() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}

    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_code"; then
        echo -e "${GREEN}✅ $name${NC} - OK"
        return 0
    else
        echo -e "${RED}❌ $name${NC} - FAILED"
        return 1
    fi
}

# Function to check docker container
check_container() {
    local name=$1

    if docker compose ps "$name" | grep -q "Up"; then
        echo -e "${GREEN}✅ $name${NC} - Running"
        return 0
    else
        echo -e "${RED}❌ $name${NC} - Not running"
        return 1
    fi
}

# Check containers
echo "📦 Checking containers..."
check_container "ollama"
check_container "qdrant"
check_container "postgres"
check_container "mongodb"
check_container "openmemory"
check_container "rag_api"
check_container "reranker"
check_container "librechat"
check_container "nginx"
check_container "redis"

echo ""
echo "🌐 Checking HTTP endpoints..."
check_http "Nginx" "http://localhost:80/health"
check_http "LibreChat" "http://localhost:3080/api/health"
check_http "Ollama" "http://localhost:11434/api/tags"
check_http "Qdrant" "http://localhost:6333/health"
check_http "mem0" "http://localhost:8080/health"
check_http "RAG API" "http://localhost:8000/health"
check_http "Reranker" "http://localhost:8001/health"
check_http "Redis" "http://localhost:6379" 200  # Redis doesn't have HTTP health check

echo ""
echo "🗄️  Checking databases..."

# PostgreSQL
if docker compose exec -T postgres pg_isready -U librechat > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL${NC} - Ready"
else
    echo -e "${RED}❌ PostgreSQL${NC} - Not ready"
fi

# MongoDB
if docker compose exec -T mongodb mongosh --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ MongoDB${NC} - Ready"
else
    echo -e "${RED}❌ MongoDB${NC} - Not ready"
fi

echo ""
echo "📊 Checking Qdrant collections..."

# Check collections exist
collections=$(curl -s http://localhost:6333/collections | python3 -c "import sys, json; print(len(json.load(sys.stdin)['result']['collections']))")

if [ "$collections" -ge 3 ]; then
    echo -e "${GREEN}✅ Qdrant Collections${NC} - $collections collections found"
else
    echo -e "${YELLOW}⚠️  Qdrant Collections${NC} - Only $collections collections found (expected 3+)"
fi

echo ""
echo "🤖 Checking Ollama models..."

# Check models
models=$(docker compose exec -T ollama ollama list | grep -c ":" || true)

if [ "$models" -ge 3 ]; then
    echo -e "${GREEN}✅ Ollama Models${NC} - $models models installed"
else
    echo -e "${YELLOW}⚠️  Ollama Models${NC} - Only $models models installed (expected 3)"
    echo -e "${YELLOW}   Run: ./scripts/pull-models.sh${NC}"
fi

echo ""
echo "💾 Checking disk usage..."
docker system df

echo ""
echo "📈 Resource usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo ""
echo "✅ Health check complete!"
