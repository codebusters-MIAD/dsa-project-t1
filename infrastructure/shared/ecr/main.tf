locals {
  repository_name = var.repository_name
  version         = trimspace(file("${path.module}/../../../${var.version_file_path}"))
  image_tag       = "${aws_ecr_repository.this.repository_url}:${local.version}"
  latest_tag      = "${aws_ecr_repository.this.repository_url}:latest"
}

resource "aws_ecr_repository" "this" {
  name         = local.repository_name
  force_delete = var.force_delete

  lifecycle {
    ignore_changes = [image_scanning_configuration]
  }

  tags = {
    Name        = local.repository_name
    Project     = var.tags.Project
    BudgetId    = var.tags.BudgetId
    Environment = var.environment
  }
}

resource "aws_ecr_lifecycle_policy" "this" {
  repository = aws_ecr_repository.this.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Expire untagged images older than 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "null_resource" "build_and_push" {
  triggers = {
    dockerfile = md5(file("${path.module}/../../../${var.dockerfile_path}/${var.dockerfile_name}"))
    version    = local.version
  }

  provisioner "local-exec" {
    command = <<EOF
      cd ${path.module}/../../..
      aws ecr get-login-password --region ${var.aws_region} | \
        docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
      
      docker build \
        -f ${var.dockerfile_path}/${var.dockerfile_name} \
        -t ${local.image_tag} \
        -t ${local.latest_tag} \
        ${var.build_context}
      
      docker push ${local.image_tag}
      docker push ${local.latest_tag}
    EOF
  }

  depends_on = [aws_ecr_repository.this]
}
