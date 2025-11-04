# GitHub Upload Guide for Colgate Kibble Segmentation Project

## 步骤 1: 初始化 Git 仓库

```bash
cd E:\Colgate
git init
```

## 步骤 2: 添加文件到 Git

```bash
git add .
```

## 步骤 3: 提交文件

```bash
git commit -m "Initial commit: Kibble pet food segmentation and super-resolution pipeline"
```

## 步骤 4: 在 GitHub 上创建新仓库

1. 登录 GitHub
2. 点击右上角的 "+" 按钮，选择 "New repository"
3. 填写仓库名称（例如：`colgate-kibble-segmentation`）
4. 选择 Public 或 Private
5. **不要**勾选 "Initialize this repository with a README"（因为我们已经有了）
6. 点击 "Create repository"

## 步骤 5: 连接到 GitHub 远程仓库

```bash
# 替换 YOUR_USERNAME 和 YOUR_REPO_NAME 为你的实际值
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 或者使用 SSH（如果你配置了SSH密钥）
# git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

## 步骤 6: 推送代码到 GitHub

```bash
git branch -M main
git push -u origin main
```

## 完整命令序列（复制粘贴执行）

```bash
cd E:\Colgate
git init
git add .
git commit -m "Initial commit: Kibble pet food segmentation and super-resolution pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 注意事项

- 如果 GitHub 要求认证，你可能需要：
  - 使用 Personal Access Token（推荐）
  - 或者配置 SSH 密钥
  
- 如果遇到认证问题，可以：
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```

- 首次推送可能需要输入 GitHub 用户名和密码/Token

## 后续更新

如果之后修改了代码，可以使用：
```bash
git add .
git commit -m "描述你的更改"
git push
```

