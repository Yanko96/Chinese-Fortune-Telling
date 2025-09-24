output "api_repository_url" {
  value = module.ecr.api_repository_url
}

output "app_repository_url" {
  value = module.ecr.app_repository_url
}

output "ecs_cluster_name" {
  value = module.ecs.cluster_name
}