variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "MIAD"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "development"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# S3 Bucket variables
variable "s3_bucket_name" {
  description = "Name of the S3 bucket for DVC storage"
  default     = "dvc-filmlens-data-repo"
  type        = string
}

variable "s3_prevent_destroy" {
  description = "Prevent accidental destruction of S3 bucket"
  type        = bool
  default     = true
}
