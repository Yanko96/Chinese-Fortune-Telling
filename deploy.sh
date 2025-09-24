#!/bin/bash
# 部署中国算命师应用到 AWS

# 配置
AWS_REGION="us-east-1"
PROJECT_NAME="fortune-teller"

# 初始化 Terraform
cd terraform/environments/production
terraform init
terraform apply -auto-approve

# 获取 ECR 仓库 URL
API_REPO_URL=$(terraform output -raw api_repository_url)
APP_REPO_URL=$(terraform output -raw app_repository_url)

# 返回项目根目录
cd ../../..

# 登录到 ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $(echo $API_REPO_URL | cut -d/ -f1)

# 构建并推送 Docker 镜像
echo "构建并推送 API 镜像..."
docker build -t $API_REPO_URL:latest ./api
docker push $API_REPO_URL:latest

echo "构建并推送 App 镜像..."
docker build -t $APP_REPO_URL:latest ./app
docker push $APP_REPO_URL:latest

# 更新 ECS 服务以使用新镜像
cd terraform/environments/production
terraform apply -auto-approve

echo "部署完成!"
echo "API URL 将在几分钟后可用，请查看 AWS ECS 控制台获取详细信息。"