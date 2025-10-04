variable "project_name" {
  description = "Project name to use in resource names"
  type        = string
}

variable "api_image_url" {
  description = "ECR URL for the API image"
  type        = string
}

variable "app_image_url" {
  description = "ECR URL for the App image"
  type        = string
}

variable "api_image_tag" {
  description = "Tag for the API image (use immutable tags like git SHA)"
  type        = string
  default     = "latest"
}

variable "app_image_tag" {
  description = "Tag for the App image (use immutable tags like git SHA)"
  type        = string
  default     = "latest"
}

variable "api_cpu" {
  description = "CPU units for API container"
  type        = string
  default     = "256"
}

variable "api_memory" {
  description = "Memory for API container"
  type        = string
  default     = "512"
}

variable "app_cpu" {
  description = "CPU units for App container"
  type        = string
  default     = "256"
}

variable "app_memory" {
  description = "Memory for App container"
  type        = string
  default     = "512"
}

variable "api_desired_count" {
  description = "Desired number of API containers"
  type        = number
  default     = 1
}

variable "app_desired_count" {
  description = "Desired number of App containers"
  type        = number
  default     = 1
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS services"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID for ECS services"
  type        = string
}

variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "api_environment_variables" {
  description = "Environment variables for the API container"
  type        = list(object({
    name  = string
    value = string
  }))
  default     = []
}

variable "app_environment_variables" {
  description = "Environment variables for the App container"
  type        = list(object({
    name  = string
    value = string
  }))
  default     = []
}

# Optional override for the App's API base URL. If unset, defaults to
# the Cloud Map private DNS name http://api.<namespace>:8000
variable "app_api_url" {
  description = "Base URL the App uses to call the API (e.g., http://<alb_dns_name>/api)"
  type        = string
  default     = ""
}

# 添加服务发现命名空间相关变量
variable "service_discovery_namespace_id" {
  description = "The ID of the service discovery namespace"
  type        = string
}

variable "service_discovery_namespace_name" {
  description = "The name of the service discovery namespace"
  type        = string
}

variable "api_target_group_arn" {
  description = "ARN of the ALB target group for API service"
  type        = string
  default     = ""
}

variable "app_target_group_arn" {
  description = "ARN of the ALB target group for App service"
  type        = string
  default     = ""
}