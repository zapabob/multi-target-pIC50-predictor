#!/bin/bash
set -e

# DAT Activity Predictor + TxGemma AI - Production Deployment Script
# Usage: ./scripts/deploy.sh [environment] [action]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
ACTION=${2:-deploy}
PROJECT_NAME="dat-predictor"
DOCKER_COMPOSE_FILE="docker-compose.yml"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deploy.log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check NVIDIA Docker (for GPU support)
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        warning "NVIDIA Docker not available - GPU features will be disabled"
    fi
    
    # Check disk space
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 50 ]; then
        error "Insufficient disk space (need at least 50GB, have ${available_space}GB)"
    fi
    
    success "Prerequisites check passed"
}

setup_environment() {
    log "Setting up environment: $ENVIRONMENT"
    
    # Create necessary directories
    mkdir -p {data,models,logs,cache,config,backups,scripts}
    
    # Set up environment file
    if [ ! -f ".env" ]; then
        log "Creating .env file..."
        cat > .env << EOF
# DAT Activity Predictor + TxGemma AI Environment Configuration
ENV=$ENVIRONMENT
COMPOSE_PROJECT_NAME=$PROJECT_NAME

# Database
POSTGRES_DB=dat_predictor
POSTGRES_USER=dat_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)

# Security
SECRET_KEY=$(openssl rand -base64 64)

# Monitoring
SENTRY_DSN=${SENTRY_DSN:-}

# GPU
CUDA_VISIBLE_DEVICES=0
EOF
        success ".env file created"
    else
        log ".env file already exists"
    fi
    
    # Set up configuration
    if [ ! -f "config/config.yaml" ]; then
        log "Creating configuration file..."
        cp config/config.yaml.example config/config.yaml
        success "Configuration file created"
    else
        log "Configuration file already exists"
    fi
    
    success "Environment setup completed"
}

backup_data() {
    log "Creating backup..."
    
    backup_timestamp=$(date +'%Y%m%d_%H%M%S')
    backup_path="$BACKUP_DIR/backup_$backup_timestamp"
    
    mkdir -p "$backup_path"
    
    # Backup data
    if [ -d "data" ]; then
        cp -r data "$backup_path/"
    fi
    
    # Backup models
    if [ -d "models" ]; then
        cp -r models "$backup_path/"
    fi
    
    # Backup configuration
    if [ -f "config/config.yaml" ]; then
        cp config/config.yaml "$backup_path/"
    fi
    
    # Backup database (if running)
    if docker-compose ps postgres | grep -q "Up"; then
        log "Backing up database..."
        docker-compose exec -T postgres pg_dump -U dat_user dat_predictor > "$backup_path/database.sql"
    fi
    
    # Compress backup
    tar -czf "$backup_path.tar.gz" -C "$BACKUP_DIR" "backup_$backup_timestamp"
    rm -rf "$backup_path"
    
    success "Backup created: $backup_path.tar.gz"
}

pull_images() {
    log "Pulling Docker images..."
    
    docker-compose pull
    
    success "Docker images pulled"
}

build_images() {
    log "Building Docker images..."
    
    docker-compose build --no-cache
    
    success "Docker images built"
}

start_services() {
    log "Starting services..."
    
    # Start infrastructure services first
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 10
    
    # Start Ollama
    docker-compose up -d ollama
    
    # Wait for Ollama to be ready
    log "Waiting for Ollama to be ready..."
    sleep 30
    
    # Start main application
    docker-compose up -d dat-predictor
    
    # Start monitoring services
    docker-compose up -d prometheus grafana
    
    # Start nginx
    docker-compose up -d nginx
    
    success "Services started"
}

stop_services() {
    log "Stopping services..."
    
    docker-compose down
    
    success "Services stopped"
}

restart_services() {
    log "Restarting services..."
    
    docker-compose restart
    
    success "Services restarted"
}

update_services() {
    log "Updating services..."
    
    # Pull latest images
    pull_images
    
    # Stop services
    stop_services
    
    # Start services
    start_services
    
    success "Services updated"
}

check_health() {
    log "Checking service health..."
    
    # Wait for services to start
    sleep 30
    
    # Check main application
    if curl -f http://localhost:8000/health &> /dev/null; then
        success "Main application is healthy"
    else
        error "Main application health check failed"
    fi
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U dat_user &> /dev/null; then
        success "Database is healthy"
    else
        error "Database health check failed"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
        success "Redis is healthy"
    else
        error "Redis health check failed"
    fi
    
    # Check Ollama
    if curl -f http://localhost:11434/api/tags &> /dev/null; then
        success "Ollama is healthy"
    else
        warning "Ollama health check failed"
    fi
    
    success "All health checks passed"
}

show_logs() {
    log "Showing service logs..."
    
    docker-compose logs -f --tail=100
}

show_status() {
    log "Service status:"
    
    docker-compose ps
    
    echo ""
    log "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

cleanup() {
    log "Cleaning up..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    success "Cleanup completed"
}

rollback() {
    log "Rolling back to previous version..."
    
    # Find latest backup
    latest_backup=$(ls -t "$BACKUP_DIR"/*.tar.gz | head -n1)
    
    if [ -z "$latest_backup" ]; then
        error "No backup found for rollback"
    fi
    
    log "Using backup: $latest_backup"
    
    # Stop services
    stop_services
    
    # Restore from backup
    tar -xzf "$latest_backup" -C "$BACKUP_DIR"
    backup_dir=$(basename "$latest_backup" .tar.gz)
    
    # Restore data
    if [ -d "$BACKUP_DIR/$backup_dir/data" ]; then
        rm -rf data
        mv "$BACKUP_DIR/$backup_dir/data" .
    fi
    
    # Restore models
    if [ -d "$BACKUP_DIR/$backup_dir/models" ]; then
        rm -rf models
        mv "$BACKUP_DIR/$backup_dir/models" .
    fi
    
    # Restore configuration
    if [ -f "$BACKUP_DIR/$backup_dir/config.yaml" ]; then
        cp "$BACKUP_DIR/$backup_dir/config.yaml" config/config.yaml
    fi
    
    # Start services
    start_services
    
    success "Rollback completed"
}

show_help() {
    echo "DAT Activity Predictor + TxGemma AI Deployment Script"
    echo ""
    echo "Usage: $0 [environment] [action]"
    echo ""
    echo "Environments:"
    echo "  production  - Production environment (default)"
    echo "  staging     - Staging environment"
    echo "  development - Development environment"
    echo ""
    echo "Actions:"
    echo "  deploy      - Deploy application (default)"
    echo "  update      - Update application"
    echo "  start       - Start services"
    echo "  stop        - Stop services"
    echo "  restart     - Restart services"
    echo "  status      - Show service status"
    echo "  logs        - Show service logs"
    echo "  health      - Check service health"
    echo "  backup      - Create backup"
    echo "  rollback    - Rollback to previous version"
    echo "  cleanup     - Clean up unused resources"
    echo "  help        - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 production deploy"
    echo "  $0 staging update"
    echo "  $0 production logs"
}

# Main execution
main() {
    log "Starting deployment script..."
    log "Environment: $ENVIRONMENT"
    log "Action: $ACTION"
    
    # Create log directory
    mkdir -p logs
    
    case $ACTION in
        deploy)
            check_prerequisites
            setup_environment
            backup_data
            build_images
            start_services
            check_health
            success "Deployment completed successfully"
            ;;
        update)
            check_prerequisites
            backup_data
            update_services
            check_health
            success "Update completed successfully"
            ;;
        start)
            start_services
            check_health
            success "Services started successfully"
            ;;
        stop)
            stop_services
            success "Services stopped successfully"
            ;;
        restart)
            restart_services
            check_health
            success "Services restarted successfully"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        health)
            check_health
            ;;
        backup)
            backup_data
            ;;
        rollback)
            rollback
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown action: $ACTION. Use 'help' for usage information."
            ;;
    esac
}

# Run main function
main "$@"
