#!/bin/bash

# Weather Data Pipeline - Production Deployment Script
# This script deploys the production environment with security and performance optimizations

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/podman-compose.prod.yml"
SECRETS_DIR="$PROJECT_ROOT/secrets"
BACKUP_DIR="$PROJECT_ROOT/backups"

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

# Check prerequisites for production deployment
check_prerequisites() {
    log_info "Checking production prerequisites..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons."
        exit 1
    fi
    
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
    
    # Check if secrets directory exists
    if [[ ! -d "$SECRETS_DIR" ]]; then
        log_error "Secrets directory not found. Please run setup-secrets.sh first."
        exit 1
    fi
    
    # Check if required secret files exist
    local required_secrets=("db_password.txt" "weatherapi_key.txt" "openweather_key.txt")
    for secret in "${required_secrets[@]}"; do
        if [[ ! -f "$SECRETS_DIR/$secret" ]]; then
            log_error "Required secret file not found: $SECRETS_DIR/$secret"
            exit 1
        fi
    done
    
    log_success "Production prerequisites checked successfully"
}

# Setup SSL certificates
setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    local ssl_dir="$PROJECT_ROOT/docker/nginx/ssl"
    mkdir -p "$ssl_dir"
    
    if [[ ! -f "$ssl_dir/cert.pem" ]] || [[ ! -f "$ssl_dir/key.pem" ]]; then
        log_warning "SSL certificates not found. Generating self-signed certificates..."
        log_warning "For production, replace with proper SSL certificates from a CA."
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$ssl_dir/key.pem" \
            -out "$ssl_dir/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        chmod 600 "$ssl_dir/key.pem"
        chmod 644 "$ssl_dir/cert.pem"
        
        log_warning "Self-signed SSL certificates generated. Replace with CA-signed certificates for production."
    else
        log_info "Using existing SSL certificates"
    fi
}

# Create backup of current deployment
create_backup() {
    log_info "Creating backup of current deployment..."
    
    mkdir -p "$BACKUP_DIR"
    local backup_name="backup-$(date +%Y%m%d-%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    mkdir -p "$backup_path"
    
    # Backup database if running
    if $COMPOSE_CMD -f "$COMPOSE_FILE" ps postgres 2>/dev/null | grep -q "Up"; then
        log_info "Backing up database..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" exec -T postgres pg_dump \
            -U weather_user -d weather_data > "$backup_path/database.sql"
    fi
    
    # Backup application data
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        cp -r "$PROJECT_ROOT/data" "$backup_path/"
    fi
    
    log_success "Backup created at $backup_path"
}

# Deploy production services
deploy_production() {
    log_info "Deploying production services..."
    
    cd "$PROJECT_ROOT"
    
    # Set production version
    export VERSION="${VERSION:-latest}"
    
    # Pull/build latest images
    log_info "Building production images..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build --no-cache
    
    # Stop existing services gracefully
    if $COMPOSE_CMD -f "$COMPOSE_FILE" ps 2>/dev/null | grep -q "Up"; then
        log_info "Stopping existing services..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" down --timeout 30
    fi
    
    # Start services with production configuration
    log_info "Starting production services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d
    
    log_success "Production services deployed successfully"
}

# Health check for production deployment
production_health_check() {
    log_info "Performing production health checks..."
    
    local max_attempts=60
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        # Check if application is responding
        if curl -sSf http://localhost/health &>/dev/null; then
            log_success "Application is healthy and responding"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Application failed to become healthy"
            return 1
        fi
        
        sleep 5
        ((attempt++))
    done
    
    # Additional production checks
    log_info "Running additional production checks..."
    
    # Check SSL certificate
    if curl -sSf -k https://localhost/health &>/dev/null; then
        log_success "HTTPS endpoint is working"
    else
        log_warning "HTTPS endpoint may have issues"
    fi
    
    # Check database connectivity
    if $COMPOSE_CMD -f "$COMPOSE_FILE" exec -T postgres pg_isready -U weather_user -d weather_data &>/dev/null; then
        log_success "Database is accessible"
    else
        log_error "Database connectivity issues"
        return 1
    fi
    
    # Check Redis connectivity
    if $COMPOSE_CMD -f "$COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis is accessible"
    else
        log_error "Redis connectivity issues"
        return 1
    fi
    
    log_success "All production health checks passed"
}

# Show production status
show_production_status() {
    log_info "Production deployment status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Application URLs:"
    echo "  - Dashboard (HTTP): http://localhost (redirects to HTTPS)"
    echo "  - Dashboard (HTTPS): https://localhost"
    echo "  - Health Check: http://localhost/health"
    echo ""
    log_info "Monitoring:"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
    echo ""
    log_info "Logs:"
    echo "  - Application: $COMPOSE_CMD -f $COMPOSE_FILE logs weather-app"
    echo "  - All services: $COMPOSE_CMD -f $COMPOSE_FILE logs"
}

# Setup monitoring and alerting
setup_monitoring() {
    log_info "Setting up production monitoring..."
    
    # Create monitoring configuration if not exists
    local monitoring_config="$PROJECT_ROOT/docker/prometheus/alerts.yml"
    if [[ ! -f "$monitoring_config" ]]; then
        log_info "Creating basic alerting rules..."
        cat > "$monitoring_config" << 'EOF'
groups:
  - name: weather_pipeline_alerts
    rules:
      - alert: ApplicationDown
        expr: up{job="weather-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Weather Pipeline Application is down"
          
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL database is down"
EOF
    fi
    
    log_success "Monitoring configuration completed"
}

# Main production deployment function
main() {
    log_info "Starting Weather Data Pipeline production deployment..."
    
    check_prerequisites
    setup_ssl
    create_backup
    setup_monitoring
    deploy_production
    production_health_check
    show_production_status
    
    log_success "Production deployment completed successfully!"
    log_info "Monitor the application at https://localhost"
    log_warning "Remember to replace self-signed SSL certificates with CA-signed certificates"
    log_info "Set up proper DNS records and firewall rules for public access"
}

# Handle command line arguments
case "${1:-}" in
    --stop)
        log_info "Stopping production services..."
        cd "$PROJECT_ROOT"
        $COMPOSE_CMD -f "$COMPOSE_FILE" down --timeout 30
        log_success "Production services stopped"
        ;;
    --restart)
        log_info "Restarting production services..."
        cd "$PROJECT_ROOT"
        create_backup
        $COMPOSE_CMD -f "$COMPOSE_FILE" restart
        production_health_check
        log_success "Production services restarted"
        ;;
    --logs)
        cd "$PROJECT_ROOT"
        $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f
        ;;
    --status)
        cd "$PROJECT_ROOT"
        show_production_status
        ;;
    --backup)
        create_backup
        ;;
    --help)
        echo "Usage: $0 [OPTION]"
        echo "Deploy and manage Weather Data Pipeline production environment"
        echo ""
        echo "Options:"
        echo "  (no args)   Deploy production environment"
        echo "  --stop      Stop all production services"
        echo "  --restart   Restart all production services"
        echo "  --logs      Show and follow logs"
        echo "  --status    Show production status"
        echo "  --backup    Create backup of current deployment"
        echo "  --help      Show this help message"
        ;;
    *)
        main
        ;;
esac
