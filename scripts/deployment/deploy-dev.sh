#!/bin/bash

# Weather Data Pipeline - Development Deployment Script
# This script sets up the development environment using podman-compose

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$PROJECT_ROOT/podman-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if podman is installed
    if ! command -v podman &> /dev/null; then
        log_error "Podman is not installed. Please install podman first."
        exit 1
    fi
    
    # Check if podman-compose is available
    if ! command -v podman-compose &> /dev/null; then
        log_warning "podman-compose not found. Trying docker-compose compatibility..."
        if ! command -v docker-compose &> /dev/null; then
            log_error "Neither podman-compose nor docker-compose found. Please install one of them."
            exit 1
        fi
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="podman-compose"
    fi
    
    log_success "Prerequisites checked successfully"
}

# Create environment file if it doesn't exist
setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating .env file from template..."
        cat > "$ENV_FILE" << 'EOF'
# Weather Data Pipeline Environment Configuration

# Environment
WEATHER_ENV=development
LOG_LEVEL=DEBUG
LOG_FORMAT=json

# API Keys (replace with your actual keys)
WEATHER_API_WEATHERAPI_KEY=your_weatherapi_key_here
WEATHER_API_OPENWEATHER_API_KEY=your_openweather_key_here

# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=weather_data
DB_USER=weather_user
DB_PASSWORD=weather_pass

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Application
DATA_DIR=/app/data
EOF
        log_warning "Please edit $ENV_FILE and add your API keys"
    else
        log_info "Using existing .env file"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$PROJECT_ROOT"/{data,logs,secrets}
    mkdir -p "$PROJECT_ROOT/data"/{raw,processed,cache,exports}
    
    log_success "Directories created successfully"
}

# Build and start services
deploy_services() {
    log_info "Building and starting services..."
    
    cd "$PROJECT_ROOT"
    
    # Pull/build images
    log_info "Building application image..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build
    
    # Start services
    log_info "Starting services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d
    
    log_success "Services started successfully"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        # Check if all services are healthy
        if $COMPOSE_CMD -f "$COMPOSE_FILE" ps | grep -q "healthy"; then
            log_success "All services are healthy"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    log_error "Services did not become healthy within expected time"
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
    return 1
}

# Show service status
show_status() {
    log_info "Service status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Application URLs:"
    echo "  - Dashboard: http://localhost:8050"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
    echo ""
    log_info "Database connections:"
    echo "  - PostgreSQL: localhost:5432 (weather_user/weather_pass)"
    echo "  - Redis: localhost:6379"
}

# Main deployment function
main() {
    log_info "Starting Weather Data Pipeline development deployment..."
    
    check_prerequisites
    setup_environment
    create_directories
    deploy_services
    wait_for_services
    show_status
    
    log_success "Development environment deployed successfully!"
    log_info "Run 'podman-compose -f $COMPOSE_FILE logs -f' to view logs"
    log_info "Run '$0 --stop' to stop all services"
}

# Handle command line arguments
case "${1:-}" in
    --stop)
        log_info "Stopping all services..."
        cd "$PROJECT_ROOT"
        $COMPOSE_CMD -f "$COMPOSE_FILE" down
        log_success "All services stopped"
        ;;
    --restart)
        log_info "Restarting all services..."
        cd "$PROJECT_ROOT"
        $COMPOSE_CMD -f "$COMPOSE_FILE" restart
        log_success "All services restarted"
        ;;
    --logs)
        cd "$PROJECT_ROOT"
        $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f
        ;;
    --status)
        cd "$PROJECT_ROOT"
        show_status
        ;;
    --help)
        echo "Usage: $0 [OPTION]"
        echo "Deploy and manage Weather Data Pipeline development environment"
        echo ""
        echo "Options:"
        echo "  (no args)   Deploy development environment"
        echo "  --stop      Stop all services"
        echo "  --restart   Restart all services"
        echo "  --logs      Show and follow logs"
        echo "  --status    Show service status"
        echo "  --help      Show this help message"
        ;;
    *)
        main
        ;;
esac
