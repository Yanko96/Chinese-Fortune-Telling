output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "security_group_id" {
  value = aws_security_group.ecs_sg.id
}

# 添加服务发现命名空间输出
output "service_discovery_namespace_id" {
  value = aws_service_discovery_private_dns_namespace.main.id
}

output "service_discovery_namespace_name" {
  value = aws_service_discovery_private_dns_namespace.main.name
}