provider "aws" {
  region = var.aws_region
}

module "ecr" {
  source = "../../modules/ecr"

  api_repository_name = "${var.project_name}-api"
  app_repository_name = "${var.project_name}-app"
}

module "networking" {
  source = "../../modules/networking"

  project_name        = var.project_name
  vpc_cidr            = var.vpc_cidr
  public_subnet_cidrs = var.public_subnet_cidrs
  availability_zones  = var.availability_zones
}

module "ecs" {
  source = "../../modules/ecs"

  project_name             = var.project_name
  api_image_url            = module.ecr.api_repository_url
  app_image_url            = module.ecr.app_repository_url
  subnet_ids               = module.networking.public_subnet_ids
  security_group_id        = module.networking.security_group_id
  aws_region               = var.aws_region
  api_environment_variables = [
    {
      name  = "GOOGLE_API_KEY"
      value = var.google_api_key
    }
  ]
  app_environment_variables = [
    {
      name  = "API_URL"
      value = "http://fortune-api:8000"
    }
  ]
}