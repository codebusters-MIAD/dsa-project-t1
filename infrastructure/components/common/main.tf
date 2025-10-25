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