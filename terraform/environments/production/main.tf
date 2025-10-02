provider "aws" {
  region = var.aws_region
}

terraform {
  backend "s3" {
    bucket         = "fortune-teller-terraform-state"  # 使用你实际创建的存储桶名称
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "fortune-teller-terraform-lock"
    encrypt        = true
  }
}

module "ecr" {
  source = "../../modules/ecr"

  api_repository_name = "${var.project_name}-api"
  app_repository_name = "${var.project_name}-app"
}

module "networking" {
  source = "../../modules/networking"

  project_name           = var.project_name
  vpc_cidr              = var.vpc_cidr
  public_subnet_cidrs   = var.public_subnet_cidrs
  availability_zones    = var.availability_zones
  alb_security_group_ids = [module.alb.alb_security_group_id]
}

module "alb" {
  source = "../../modules/alb"

  project_name      = var.project_name
  vpc_id           = module.networking.vpc_id
  public_subnet_ids = module.networking.public_subnet_ids
}

module "ecs" {
  source = "../../modules/ecs"

  project_name                     = var.project_name
  api_image_url                    = module.ecr.api_repository_url
  app_image_url                    = module.ecr.app_repository_url
  subnet_ids                       = module.networking.public_subnet_ids
  security_group_id                = module.networking.security_group_id
  aws_region                       = var.aws_region
  service_discovery_namespace_id   = module.networking.service_discovery_namespace_id
  service_discovery_namespace_name = module.networking.service_discovery_namespace_name
  api_target_group_arn             = module.alb.api_target_group_arn
  app_target_group_arn             = module.alb.app_target_group_arn
  api_environment_variables = [
    {
      name  = "GOOGLE_API_KEY"
      value = var.google_api_key
    },
    {
      name  = "API_ROOT_PATH"
      value = "/api"
    }
  ]
  app_environment_variables = []
}