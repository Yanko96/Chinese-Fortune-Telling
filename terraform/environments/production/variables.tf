variable "project_name" {
  description = "Project name to use in resource names"
  type        = string
  default     = "fortune-teller"
}

variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "availability_zones" {
  description = "Availability zones to use"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "moonshot_api_key" {
  description = "Moonshot AI (Kimi) API Key"
  type        = string
  sensitive   = true
}

variable "api_image_tag" {
  description = "Tag for the API image to deploy (e.g., git SHA or 'latest')"
  type        = string
  default     = "latest"
}

variable "app_image_tag" {
  description = "Tag for the App image to deploy (e.g., git SHA or 'latest')"
  type        = string
  default     = "latest"
}

variable "api_cpu" {
  description = "CPU units for API task (e.g., '256', '512', '1024')"
  type        = string
  default     = "256"
}

variable "api_memory" {
  description = "Memory (MiB) for API task (e.g., '512','1024','2048')"
  type        = string
  default     = "1024"
}