variable "api_repository_name" {
  description = "Name for the API ECR repository"
  type        = string
  default     = "fortune-teller-api"
}

variable "app_repository_name" {
  description = "Name for the App ECR repository"
  type        = string
  default     = "fortune-teller-app"
}