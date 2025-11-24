
# S3 Bucket Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for DVC storage"
  value       = module.s3_bucket.bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = module.s3_bucket.bucket_arn
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = module.s3_bucket.bucket_domain_name
}

# DVC Configuration
output "dvc_remote_config_command" {
  description = "Command to configure DVC remote storage"
  value       = "dvc remote add -d storage s3://${module.s3_bucket.bucket_name}"
}

# ECR API Outputs
output "ecr_api_repository_url" {
  description = "URL of the API ECR repository"
  value       = module.ecr_api.repository_url
}

output "ecr_api_repository_name" {
  description = "Name of the API ECR repository"
  value       = module.ecr_api.repository_name
}

output "ecr_api_image_tag" {
  description = "Full image tag for API"
  value       = module.ecr_api.image_tag
}

output "ecr_api_version" {
  description = "Current API version"
  value       = module.ecr_api.version
}

# ECR Query API Outputs
output "ecr_query_api_repository_url" {
  description = "URL of the Query API ECR repository"
  value       = module.ecr_query_api.repository_url
}

output "ecr_query_api_repository_name" {
  description = "Name of the Query API ECR repository"
  value       = module.ecr_query_api.repository_name
}

output "ecr_query_api_image_tag" {
  description = "Full image tag for Query API"
  value       = module.ecr_query_api.image_tag
}

output "ecr_query_api_version" {
  description = "Current Query API version"
  value       = module.ecr_query_api.version
}