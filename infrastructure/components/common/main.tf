# Create S3 Bucket for DVC storage
module "s3_bucket" {
  source = "../../shared/s3"

  bucket_name       = var.s3_bucket_name
  enable_versioning = true
  prevent_destroy   = var.s3_prevent_destroy

  tags = merge(
    local.common_tags,
    {
      Name    = var.s3_bucket_name
      Purpose = "DVC Remote Storage"
    }
  )
}

# ECR Repository for API Service
module "ecr_api" {
  source = "../../shared/ecr"

  environment       = var.environment
  aws_region        = var.aws_region
  repository_name   = "${lower(var.project_name)}-api"
  force_delete      = var.environment != "prod"
  
  dockerfile_path   = "docker"
  dockerfile_name   = "Dockerfile.api"
  version_file_path = "src/api/VERSION"
  build_context     = "."

  tags = local.common_tags
}

# ECR Repository for Query API Service
module "ecr_query_api" {
  source = "../../shared/ecr"

  environment       = var.environment
  aws_region        = var.aws_region
  repository_name   = "${lower(var.project_name)}-query-api"
  force_delete      = var.environment != "prod"
  
  dockerfile_path   = "docker"
  dockerfile_name   = "Dockerfile.query-api"
  version_file_path = "src/query_api/VERSION"
  build_context     = "."

  tags = local.common_tags
}

# ECR Repository for MLflow Service
module "ecr_mlflow" {
  source = "../../shared/ecr"

  environment       = var.environment
  aws_region        = var.aws_region
  repository_name   = "${lower(var.project_name)}-mlflow"
  force_delete      = var.environment != "prod"
  
  dockerfile_path   = "docker"
  dockerfile_name   = "Dockerfile.mlflow"
  version_file_path = null  # MLflow uses base image version
  build_context     = "."

  tags = local.common_tags
}