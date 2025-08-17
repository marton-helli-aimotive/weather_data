#!/bin/bash

# Weather Data Pipeline - Secrets Setup Script
# This script helps set up secrets for production deployment securely

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECRETS_DIR="$PROJECT_ROOT/secrets"

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

# Create secrets directory with proper permissions
create_secrets_directory() {
    log_info "Creating secrets directory..."
    
    if [[ ! -d "$SECRETS_DIR" ]]; then
        mkdir -p "$SECRETS_DIR"
        chmod 700 "$SECRETS_DIR"
        log_success "Secrets directory created with restricted permissions"
    else
        log_info "Secrets directory already exists"
        chmod 700 "$SECRETS_DIR"
    fi
}

# Generate secure random password
generate_password() {
    local length="${1:-32}"
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length"
}

# Setup database password
setup_db_password() {
    local password_file="$SECRETS_DIR/db_password.txt"
    
    if [[ ! -f "$password_file" ]]; then
        log_info "Generating database password..."
        
        read -p "Enter database password (or press Enter to generate): " -s db_password
        echo
        
        if [[ -z "$db_password" ]]; then
            db_password=$(generate_password 32)
            log_info "Generated secure database password"
        fi
        
        echo -n "$db_password" > "$password_file"
        chmod 600 "$password_file"
        log_success "Database password saved securely"
    else
        log_info "Database password already exists"
    fi
}

# Setup WeatherAPI key
setup_weatherapi_key() {
    local key_file="$SECRETS_DIR/weatherapi_key.txt"
    
    if [[ ! -f "$key_file" ]]; then
        log_info "Setting up WeatherAPI key..."
        
        echo "Get your WeatherAPI key from: https://www.weatherapi.com/signup.aspx"
        read -p "Enter your WeatherAPI key: " weatherapi_key
        
        if [[ -n "$weatherapi_key" ]]; then
            echo -n "$weatherapi_key" > "$key_file"
            chmod 600 "$key_file"
            log_success "WeatherAPI key saved securely"
        else
            echo -n "placeholder" > "$key_file"
            chmod 600 "$key_file"
            log_warning "Placeholder WeatherAPI key created. Please update $key_file with your actual key."
        fi
    else
        log_info "WeatherAPI key already exists"
    fi
}

# Setup OpenWeatherMap API key
setup_openweather_key() {
    local key_file="$SECRETS_DIR/openweather_key.txt"
    
    if [[ ! -f "$key_file" ]]; then
        log_info "Setting up OpenWeatherMap API key..."
        
        echo "Get your OpenWeatherMap API key from: https://openweathermap.org/api"
        read -p "Enter your OpenWeatherMap API key (optional): " openweather_key
        
        if [[ -n "$openweather_key" ]]; then
            echo -n "$openweather_key" > "$key_file"
            chmod 600 "$key_file"
            log_success "OpenWeatherMap API key saved securely"
        else
            echo -n "optional" > "$key_file"
            chmod 600 "$key_file"
            log_info "Optional OpenWeatherMap API key placeholder created"
        fi
    else
        log_info "OpenWeatherMap API key already exists"
    fi
}

# Setup JWT secret for authentication
setup_jwt_secret() {
    local secret_file="$SECRETS_DIR/jwt_secret.txt"
    
    if [[ ! -f "$secret_file" ]]; then
        log_info "Generating JWT secret..."
        
        local jwt_secret=$(generate_password 64)
        echo -n "$jwt_secret" > "$secret_file"
        chmod 600 "$secret_file"
        log_success "JWT secret generated and saved securely"
    else
        log_info "JWT secret already exists"
    fi
}

# Setup encryption key for sensitive data
setup_encryption_key() {
    local key_file="$SECRETS_DIR/encryption_key.txt"
    
    if [[ ! -f "$key_file" ]]; then
        log_info "Generating encryption key..."
        
        local encryption_key=$(generate_password 32)
        echo -n "$encryption_key" > "$key_file"
        chmod 600 "$key_file"
        log_success "Encryption key generated and saved securely"
    else
        log_info "Encryption key already exists"
    fi
}

# Validate all secrets
validate_secrets() {
    log_info "Validating secrets..."
    
    local secrets=(
        "db_password.txt"
        "weatherapi_key.txt"
        "openweather_key.txt"
        "jwt_secret.txt"
        "encryption_key.txt"
    )
    
    local all_valid=true
    
    for secret in "${secrets[@]}"; do
        local secret_file="$SECRETS_DIR/$secret"
        
        if [[ -f "$secret_file" ]]; then
            local permissions=$(stat -c %a "$secret_file" 2>/dev/null || stat -f %A "$secret_file" 2>/dev/null || echo "unknown")
            
            if [[ "$permissions" != "600" ]]; then
                log_warning "Secret file $secret has incorrect permissions: $permissions (should be 600)"
                chmod 600 "$secret_file"
                log_info "Fixed permissions for $secret"
            fi
            
            local content=$(cat "$secret_file")
            if [[ -z "$content" ]] || [[ "$content" == "placeholder" ]] || [[ "$content" == "optional" ]]; then
                if [[ "$secret" == "openweather_key.txt" && "$content" == "optional" ]]; then
                    log_info "OpenWeatherMap API key is optional"
                else
                    log_error "Secret file $secret is empty or contains placeholder"
                    all_valid=false
                fi
            else
                log_success "Secret $secret is valid"
            fi
        else
            log_error "Secret file $secret is missing"
            all_valid=false
        fi
    done
    
    if [[ "$all_valid" == "true" ]]; then
        log_success "All secrets are valid and properly configured"
    else
        log_error "Some secrets need attention before production deployment"
        return 1
    fi
}

# Show secrets status
show_secrets_status() {
    log_info "Secrets status:"
    
    local secrets=(
        "db_password.txt:Database Password"
        "weatherapi_key.txt:WeatherAPI Key"
        "openweather_key.txt:OpenWeatherMap Key"
        "jwt_secret.txt:JWT Secret"
        "encryption_key.txt:Encryption Key"
    )
    
    for secret_info in "${secrets[@]}"; do
        IFS=':' read -r filename description <<< "$secret_info"
        local secret_file="$SECRETS_DIR/$filename"
        
        if [[ -f "$secret_file" ]]; then
            local content=$(cat "$secret_file")
            if [[ -n "$content" && "$content" != "placeholder" && "$content" != "optional" ]]; then
                echo "  ✓ $description: Configured"
            elif [[ "$content" == "optional" ]]; then
                echo "  ? $description: Optional (not configured)"
            else
                echo "  ✗ $description: Needs configuration"
            fi
        else
            echo "  ✗ $description: Missing"
        fi
    done
}

# Backup secrets
backup_secrets() {
    if [[ ! -d "$SECRETS_DIR" ]]; then
        log_error "Secrets directory does not exist"
        return 1
    fi
    
    local backup_file="$PROJECT_ROOT/secrets-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    log_info "Creating encrypted backup of secrets..."
    
    tar -czf - -C "$PROJECT_ROOT" secrets | gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 --s2k-digest-algo SHA512 --s2k-count 65536 --symmetric --output "$backup_file"
    
    if [[ $? -eq 0 ]]; then
        log_success "Encrypted secrets backup created: $backup_file"
        log_warning "Store this backup file securely and remember the passphrase"
    else
        log_error "Failed to create secrets backup"
        return 1
    fi
}

# Restore secrets from backup
restore_secrets() {
    read -p "Enter the path to the encrypted backup file: " backup_file
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file does not exist: $backup_file"
        return 1
    fi
    
    log_info "Restoring secrets from encrypted backup..."
    
    if [[ -d "$SECRETS_DIR" ]]; then
        log_warning "Existing secrets directory will be backed up"
        mv "$SECRETS_DIR" "$SECRETS_DIR.backup.$(date +%Y%m%d-%H%M%S)"
    fi
    
    gpg --decrypt "$backup_file" | tar -xzf - -C "$PROJECT_ROOT"
    
    if [[ $? -eq 0 ]]; then
        log_success "Secrets restored successfully"
        validate_secrets
    else
        log_error "Failed to restore secrets from backup"
        return 1
    fi
}

# Main setup function
main() {
    log_info "Setting up secrets for Weather Data Pipeline..."
    
    create_secrets_directory
    setup_db_password
    setup_weatherapi_key
    setup_openweather_key
    setup_jwt_secret
    setup_encryption_key
    validate_secrets
    
    log_success "Secrets setup completed successfully!"
    echo ""
    show_secrets_status
    echo ""
    log_info "Secrets are stored in: $SECRETS_DIR"
    log_warning "Keep these secrets secure and never commit them to version control"
    log_info "Use '$0 --backup' to create an encrypted backup of secrets"
}

# Handle command line arguments
case "${1:-}" in
    --validate)
        validate_secrets
        ;;
    --status)
        show_secrets_status
        ;;
    --backup)
        backup_secrets
        ;;
    --restore)
        restore_secrets
        ;;
    --help)
        echo "Usage: $0 [OPTION]"
        echo "Set up and manage secrets for Weather Data Pipeline"
        echo ""
        echo "Options:"
        echo "  (no args)   Set up all secrets interactively"
        echo "  --validate  Validate existing secrets"
        echo "  --status    Show secrets status"
        echo "  --backup    Create encrypted backup of secrets"
        echo "  --restore   Restore secrets from encrypted backup"
        echo "  --help      Show this help message"
        ;;
    *)
        main
        ;;
esac
