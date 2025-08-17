# Deployment Guide

This guide covers the complete deployment process for the Weather Data Pipeline, from local development to production deployment using Podman containers.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, RHEL 8+, or equivalent)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for production)
- **Storage**: Minimum 20GB free space (100GB+ recommended for production)
- **CPU**: 2+ cores (4+ cores recommended for production)

### Software Dependencies

1. **Podman** (4.0+) or Docker (20.10+)
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y podman
   
   # RHEL/CentOS
   sudo dnf install -y podman
   ```

2. **Podman Compose** or Docker Compose
   ```bash
   # Install podman-compose
   pip install podman-compose
   
   # Or use docker-compose as fallback
   sudo apt-get install -y docker-compose
   ```

3. **OpenSSL** (for SSL certificate generation)
   ```bash
   sudo apt-get install -y openssl
   ```

### API Keys

Obtain API keys from the following services:

1. **WeatherAPI** (Required)
   - Sign up at: https://www.weatherapi.com/signup.aspx
   - Free tier: 1 million requests/month

2. **OpenWeatherMap** (Optional)
   - Sign up at: https://openweathermap.org/api
   - Free tier: 1000 requests/day

## Development Deployment

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd weather_data
   ```

2. **Run the development deployment script**:
   ```bash
   ./scripts/deployment/deploy-dev.sh
   ```

3. **Access the application**:
   - Dashboard: http://localhost:8050
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### Manual Development Setup

1. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

2. **Start services**:
   ```bash
   podman-compose up -d
   ```

3. **Check service health**:
   ```bash
   podman-compose ps
   ./scripts/deployment/deploy-dev.sh --status
   ```

### Development Environment Services

| Service | Port | Description |
|---------|------|-------------|
| weather-app | 8050 | Main application dashboard |
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Redis cache |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboards |

## Production Deployment

### Pre-deployment Checklist

1. **Set up secrets**:
   ```bash
   ./scripts/deployment/setup-secrets.sh
   ```

2. **Verify system requirements**:
   - Sufficient disk space
   - Network connectivity
   - Firewall configuration
   - DNS records (if applicable)

3. **Configure SSL certificates**:
   - Place SSL certificates in `docker/nginx/ssl/`
   - Or use the auto-generated self-signed certificates for testing

### Production Deployment Process

1. **Deploy production environment**:
   ```bash
   ./scripts/deployment/deploy-prod.sh
   ```

2. **Verify deployment**:
   ```bash
   ./scripts/deployment/deploy-prod.sh --status
   ```

3. **Access the application**:
   - HTTP: http://your-domain.com (redirects to HTTPS)
   - HTTPS: https://your-domain.com

### Production Environment Services

| Service | Internal Port | External Port | Description |
|---------|---------------|---------------|-------------|
| nginx | 80, 443 | 80, 443 | Reverse proxy and SSL termination |
| weather-app | 8050 | - | Main application (behind proxy) |
| postgres | 5432 | - | PostgreSQL database (internal only) |
| redis | 6379 | - | Redis cache (internal only) |
| prometheus | 9090 | - | Metrics collection |
| grafana | 3000 | - | Monitoring dashboards |
| fluentd | 24224 | - | Log aggregation |

### Production Configuration

#### Environment Variables

Production-specific environment variables:

```bash
WEATHER_ENV=production
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
JWT_SECRET_FILE=/run/secrets/jwt_secret
ENCRYPTION_KEY_FILE=/run/secrets/encryption_key

# Database
DB_PASSWORD_FILE=/run/secrets/db_password

# API Keys
WEATHER_API_WEATHERAPI_KEY_FILE=/run/secrets/weatherapi_key
WEATHER_API_OPENWEATHER_API_KEY_FILE=/run/secrets/openweather_key
```

#### Resource Limits

Production containers have resource limits configured:

- **weather-app**: 2 CPU cores, 2GB RAM
- **postgres**: 1 CPU core, 1GB RAM
- **redis**: 0.5 CPU cores, 512MB RAM

## Monitoring and Observability

### Health Checks

The application includes comprehensive health checks:

1. **Container Health Checks**:
   ```bash
   # Check individual container health
   podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Health}}"
   ```

2. **Application Health Endpoint**:
   ```bash
   curl http://localhost/health
   ```

3. **CLI Health Check**:
   ```bash
   weather-pipeline check
   ```

### Metrics and Monitoring

#### Prometheus Metrics

Available metrics endpoints:

- Application metrics: http://localhost:8000/metrics
- Node metrics: http://localhost:9100/metrics (if node_exporter is deployed)
- Container metrics: http://localhost:8080/metrics (if cAdvisor is deployed)

#### Grafana Dashboards

Pre-configured dashboards available at http://localhost:3000:

- System Overview
- Application Performance
- Database Metrics
- API Usage Statistics

#### Log Aggregation

Logs are collected by Fluentd and can be forwarded to:

- Elasticsearch
- CloudWatch
- Splunk
- File system

### Alerting

Basic alerting rules are configured in `docker/prometheus/alerts.yml`:

- Application down
- High memory usage
- Database connectivity issues
- API rate limiting

## Security Considerations

### Container Security

1. **Non-root user**: All containers run as non-root users
2. **Read-only file systems**: Production containers use read-only root filesystems
3. **Security options**: `no-new-privileges` is enabled
4. **Resource limits**: CPU and memory limits prevent resource exhaustion

### Network Security

1. **Internal network**: Services communicate over a dedicated bridge network
2. **Port binding**: Database and cache ports are bound to localhost only
3. **SSL/TLS**: HTTPS is enforced with secure cipher configurations
4. **Rate limiting**: API endpoints have rate limiting configured

### Secrets Management

1. **Docker secrets**: Sensitive data is stored using Docker/Podman secrets
2. **File permissions**: Secret files have 600 permissions
3. **Environment isolation**: Secrets are not exposed in environment variables
4. **Encryption**: Secrets backup is encrypted using GPG

### Security Headers

The Nginx configuration includes security headers:

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`

## Troubleshooting

### Common Issues

#### 1. Container Failed to Start

```bash
# Check container logs
podman-compose logs <service-name>

# Check container status
podman ps -a

# Restart specific service
podman-compose restart <service-name>
```

#### 2. Database Connection Issues

```bash
# Check database health
podman-compose exec postgres pg_isready -U weather_user -d weather_data

# Check database logs
podman-compose logs postgres

# Reset database
podman-compose down postgres
podman volume rm weather_data_postgres_data
podman-compose up -d postgres
```

#### 3. API Key Issues

```bash
# Verify API keys
./scripts/deployment/setup-secrets.sh --status

# Test API connectivity
curl "https://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q=London"
```

#### 4. SSL Certificate Issues

```bash
# Check certificate validity
openssl x509 -in docker/nginx/ssl/cert.pem -text -noout

# Regenerate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

### Log Analysis

#### Application Logs

```bash
# Follow application logs
podman-compose logs -f weather-app

# Search for errors
podman-compose logs weather-app | grep ERROR

# Export logs
podman-compose logs weather-app > app.log
```

#### System Logs

```bash
# Check system resource usage
docker stats

# Check disk usage
df -h

# Check memory usage
free -h
```

### Performance Troubleshooting

#### Database Performance

```bash
# Check database performance
podman-compose exec postgres psql -U weather_user -d weather_data -c "
SELECT query, calls, total_time, rows, 100.0 * shared_blks_hit /
       nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5;"
```

#### Cache Performance

```bash
# Check Redis performance
podman-compose exec redis redis-cli info stats
```

## Maintenance

### Regular Maintenance Tasks

#### 1. Database Maintenance

```bash
# Create database backup
./scripts/deployment/deploy-prod.sh --backup

# Database vacuum and analyze
podman-compose exec postgres psql -U weather_user -d weather_data -c "VACUUM ANALYZE;"

# Update statistics
podman-compose exec postgres psql -U weather_user -d weather_data -c "ANALYZE;"
```

#### 2. Log Rotation

```bash
# Rotate container logs
podman system prune -f --volumes

# Clean old images
podman image prune -a
```

#### 3. Security Updates

```bash
# Update base images
podman-compose pull
podman-compose up -d --force-recreate

# Update application
git pull
podman-compose build --no-cache
podman-compose up -d
```

#### 4. Performance Monitoring

```bash
# Check system performance
htop
iotop
nethogs

# Check application metrics
curl http://localhost:9090/api/v1/query?query=up
```

### Backup and Recovery

#### Database Backup

```bash
# Create full backup
podman-compose exec postgres pg_dump -U weather_user -d weather_data > backup.sql

# Restore from backup
podman-compose exec -T postgres psql -U weather_user -d weather_data < backup.sql
```

#### Application Data Backup

```bash
# Backup application data
tar -czf app-data-$(date +%Y%m%d).tar.gz data/

# Backup configuration
tar -czf config-$(date +%Y%m%d).tar.gz docker/ scripts/ *.yml
```

#### Secrets Backup

```bash
# Create encrypted secrets backup
./scripts/deployment/setup-secrets.sh --backup

# Restore secrets from backup
./scripts/deployment/setup-secrets.sh --restore
```

### Scaling

#### Horizontal Scaling

For high-availability deployment, consider:

1. **Load balancer**: Use HAProxy or similar
2. **Database clustering**: PostgreSQL with replication
3. **Cache clustering**: Redis Cluster
4. **Container orchestration**: Kubernetes or Docker Swarm

#### Vertical Scaling

Adjust resource limits in `podman-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Increase CPU
      memory: 4G       # Increase memory
    reservations:
      cpus: '1.0'
      memory: 1G
```

## CI/CD Integration

### GitHub Actions

The included CI/CD pipeline (`.github/workflows/ci-cd.yml`) provides:

1. **Code quality checks**: Linting, formatting, type checking
2. **Security scanning**: Bandit, Trivy
3. **Testing**: Unit, integration, and performance tests
4. **Container building**: Multi-architecture builds
5. **Deployment**: Automated staging and production deployment

### Custom CI/CD

Adapt the deployment scripts for your CI/CD platform:

```bash
# Build and deploy in CI
./scripts/deployment/setup-secrets.sh --validate
./scripts/deployment/deploy-prod.sh
```

## Support

For additional support:

1. Check the [troubleshooting section](#troubleshooting)
2. Review application logs
3. Consult the monitoring dashboards
4. Create an issue in the project repository

---

**Note**: This deployment guide assumes familiarity with containerization and basic system administration. For production deployments, consider consulting with a DevOps engineer or system administrator.
