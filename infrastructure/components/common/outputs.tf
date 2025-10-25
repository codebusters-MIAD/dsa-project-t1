
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