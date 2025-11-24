# Common variables
variable "environment" {
  type        = string
  description = "Environment name (dev, staging, prod)"
}

variable "aws_region" {
  type        = string
  description = "AWS region"
}

# ECR Repository
variable "repository_name" {
  type        = string
  description = "Name of the ECR repository"
}

variable "force_delete" {
  type        = bool
  default     = false
  description = "Force delete repository even if it contains images"
}

# Docker Build Configuration
variable "dockerfile_path" {
  type        = string
  description = "Path to the directory containing the Dockerfile (relative to root)"
}

variable "version_file_path" {
  type        = string
  description = "Path to the VERSION file (relative to root) used as trigger"
}

variable "build_context" {
  type        = string
  default     = "."
  description = "Docker build context path (relative to dockerfile_path)"
}

variable "dockerfile_name" {
  type        = string
  default     = "Dockerfile"
  description = "Name of the Dockerfile"
}

# Tags
variable "tags" {
  type = object({
    Project  = string
    BudgetId = string
  })
  description = "Common tags for all resources"
}

