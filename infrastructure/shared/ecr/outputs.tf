output "repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.this.arn
}

output "repository_name" {
  description = "Name of the ECR repository"
  value       = aws_ecr_repository.this.name
}

output "image_tag" {
  description = "Full image tag with version"
  value       = local.image_tag
}

output "version" {
  description = "Current version from VERSION file"
  value       = local.version
}
