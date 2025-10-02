# ALB (Application Load Balancer) 部署指南

## 概述
此更新为项目添加了Application Load Balancer (ALB)功能，提供固定的DNS地址以替代直接IP访问。

## 新增功能

### 1. ALB模块 (`terraform/modules/alb/`)
- **Application Load Balancer**: 提供负载均衡和固定DNS名称
- **Target Groups**: 为API和Web应用分别创建目标组
- **健康检查**: 自动监控服务健康状态
- **安全组**: ALB专用安全组，允许HTTP(80)和HTTPS(443)流量

### 2. 路由配置
- **主页面**: `http://your-alb-dns-name/` → Streamlit应用 (端口8501)
- **API接口**: `http://your-alb-dns-name/api/*` → FastAPI服务 (端口8000)

### 3. 安全改进
- ECS服务现在主要通过ALB接收流量
- 临时保留直接访问规则以确保平稳过渡
- 后续可以移除直接访问规则以增强安全性

## 部署步骤

### 1. 规划和验证
```powershell
cd terraform/environments/production
terraform plan
```

### 2. 部署ALB和相关资源
```powershell
terraform apply
```

### 3. 获取ALB DNS地址
部署完成后，Terraform将输出ALB的DNS地址：
```
Outputs:
alb_dns_name = "fortune-teller-alb-xxxxxxxxx.us-east-1.elb.amazonaws.com"
application_urls = {
  "api" = "http://fortune-teller-alb-xxxxxxxxx.us-east-1.elb.amazonaws.com/api"
  "web_app" = "http://fortune-teller-alb-xxxxxxxxx.us-east-1.elb.amazonaws.com"
}
```

### 4. 验证部署
- 访问 Web应用: `http://your-alb-dns-name/`
- 测试 API: `http://your-alb-dns-name/api/health` (如果有健康检查端点)

## 配置详情

### ALB目标组配置
- **API Target Group**: 
  - 端口: 8000
  - 健康检查: `/` (HTTP 200/404)
  - 目标类型: IP
  
- **App Target Group**:
  - 端口: 8501  
  - 健康检查: `/_stcore/health` (Streamlit健康检查)
  - 目标类型: IP

### 安全组变更
- **ALB安全组**: 允许来自互联网的80和443端口访问
- **ECS安全组**: 现在主要接收来自ALB的流量，临时保留直接访问

## 后续优化建议

### 1. 移除临时直接访问（部署验证后）
在 `terraform/modules/networking/main.tf` 中移除以下规则:
```hcl
# 移除这些临时规则
ingress {
  from_port   = 8000
  to_port     = 8000
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  description = "Temporary direct API access - remove after ALB deployment"
}

ingress {
  from_port   = 8501
  to_port     = 8501
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  description = "Temporary direct App access - remove after ALB deployment"
}
```

### 2. 添加HTTPS支持
- 申请SSL证书 (AWS Certificate Manager)
- 为ALB添加HTTPS监听器
- 配置HTTP到HTTPS重定向

### 3. 添加自定义域名
- 在Route53中创建CNAME记录
- 指向ALB DNS名称

## 故障排除

### 健康检查失败
- 检查ECS任务是否正常运行
- 验证安全组规则是否正确
- 查看ALB目标组健康状态

### 无法访问应用
- 确认ALB监听器规则配置正确
- 检查目标组注册状态
- 验证ECS服务是否已注册到目标组

### DNS解析问题
- ALB DNS名称需要几分钟才能完全传播
- 使用 `nslookup` 或 `dig` 验证DNS解析

## 成本影响
添加ALB将产生额外费用:
- ALB基础费用: ~$16/月
- 负载均衡器容量单位 (LCU): 基于使用量
- 详细定价请参考AWS ALB定价页面