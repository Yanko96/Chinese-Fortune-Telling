output "cluster_name" {
  description = "The name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "api_service_name" {
  description = "The name of the API ECS service"
  value       = aws_ecs_service.api.name
}

output "app_service_name" {
  description = "The name of the App ECS service"
  value       = aws_ecs_service.app.name
}