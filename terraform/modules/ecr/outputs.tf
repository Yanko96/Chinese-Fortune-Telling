output "api_repository_url" {
  value = aws_ecr_repository.fortune_api.repository_url
}

output "app_repository_url" {
  value = aws_ecr_repository.fortune_app.repository_url
}