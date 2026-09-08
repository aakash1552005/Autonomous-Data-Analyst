#!/usr/bin/env bash
# ==============================================================================
# Autonomous Data Analyst (ADA) — Automated Cloud VPS Deployment Script
# Supports: Ubuntu 20.04 / 22.04 / 24.04 LTS, Debian 11 / 12
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN}    🚀 Autonomous Data Analyst (ADA) — Cloud VM Deployment Setup   ${NC}"
echo -e "${BLUE}=====================================================================${NC}"

# 1. Root / Sudo Check
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}[!] Not running as root. Attempting to use sudo...${NC}"
    SUDO="sudo"
else
    SUDO=""
fi

# 2. Update System Packages
echo -e "\n${BLUE}[1/6] Updating system package lists...${NC}"
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    ufw

# 3. Install Docker & Docker Compose if missing
if ! command -v docker &> /dev/null; then
    echo -e "\n${BLUE}[2/6] Installing Docker Engine...${NC}"
    $SUDO install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    $SUDO chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      $SUDO tee /etc/apt/sources.list.d/docker.list > /dev/null

    $SUDO apt-get update -y
    $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    $SUDO systemctl enable --now docker
    echo -e "${GREEN}[✔] Docker installed successfully.${NC}"
else
    echo -e "\n${GREEN}[2/6] Docker is already installed: $(docker --version)${NC}"
fi

# 4. Configure Firewall (UFW)
echo -e "\n${BLUE}[3/6] Configuring firewall rules (UFW)...${NC}"
$SUDO ufw allow 22/tcp || true      # SSH
$SUDO ufw allow 80/tcp || true      # HTTP
$SUDO ufw allow 443/tcp || true     # HTTPS
$SUDO ufw allow 8501/tcp || true    # Streamlit Web UI
$SUDO ufw allow 5000/tcp || true    # Flask REST API
$SUDO ufw allow 5678/tcp || true    # n8n Automation Engine
# Enable UFW if not active, without prompting for SSH disconnect
echo "y" | $SUDO ufw enable || true
echo -e "${GREEN}[✔] Ports 8501, 5000, 5678, 80, 443, and 22 opened.${NC}"

# 5. Directory & Environment Configuration
echo -e "\n${BLUE}[4/6] Setting up project directories and environment...${NC}"
mkdir -p runs data/sample n8n/workflows

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}[✔] Created .env from .env.example template.${NC}"
    else
        touch .env
    fi
fi

# 6. Build and Launch Containers via Docker Compose
echo -e "\n${BLUE}[5/6] Building and launching multi-service stack via Docker Compose...${NC}"
$SUDO docker compose down --remove-orphans || true
$SUDO docker compose up -d --build

# 7. Verification and Status Display
echo -e "\n${BLUE}[6/6] Verifying service health...${NC}"
sleep 8

PUBLIC_IP=$(curl -s https://api.ipify.org || curl -s https://ifconfig.me || echo "YOUR_SERVER_IP")

echo -e "\n${GREEN}=====================================================================${NC}"
echo -e "${GREEN}   🎉 Autonomous Data Analyst is LIVE and RUNNING 24/7 in Cloud!    ${NC}"
echo -e "${GREEN}=====================================================================${NC}"
echo -e "Access your live endpoints below:\n"
echo -e "  🖥️  Streamlit Dashboard : ${BLUE}http://${PUBLIC_IP}:8501${NC}"
echo -e "  ⚡  Flask REST API     : ${BLUE}http://${PUBLIC_IP}:5000/health${NC}"
echo -e "  🔄  n8n Workflow Engine: ${BLUE}http://${PUBLIC_IP}:5678${NC}"
echo -e "      ${YELLOW}(n8n default credentials: user='admin', password='changeme123')${NC}"
echo -e "\nUseful Operational Commands:"
echo -e "  - View live logs    : ${YELLOW}docker compose logs -f${NC}"
echo -e "  - Restart services  : ${YELLOW}docker compose restart${NC}"
echo -e "  - Update to latest  : ${YELLOW}git pull && docker compose up -d --build${NC}"
echo -e "  - Stop stack        : ${YELLOW}docker compose down${NC}"
echo -e "${GREEN}=====================================================================${NC}\n"
