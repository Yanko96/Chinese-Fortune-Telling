output "api_repository_url" {
  value = module.ecr.api_repository_url
}

output "app_repository_url" {
  value = module.ecr.app_repository_url
}

output "ecs_cluster_name" {
  value = module.ecs.cluster_name
}

output "alb_dns_name" {
  value = module.alb.alb_dns_name
  description = "Application Load Balancer DNS名称 - 使用此地址访问应用"
}

output "alb_zone_id" {
  value = module.alb.alb_zone_id
  description = "ALB的Route53 Hosted Zone ID"
}

output "application_urls" {
  value = {
    web_app = "http://${module.alb.alb_dns_name}"
    api     = "http://${module.alb.alb_dns_name}/api"
  }
  description = "应用访问URL"
}