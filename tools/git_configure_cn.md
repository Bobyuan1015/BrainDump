# 在 Windows 上配置 Git

## 1. 安装 Git
1. 访问 [Git 官方网站](https://git-scm.com/)。
2. 下载 Windows 版本的 Git。
3. 下载完成后运行安装程序。

## 2. 配置安装选项
在安装过程中，Git 会提示配置选项：
- **安装目录**：使用默认路径或选择自定义路径。
- **编辑器**：选择默认编辑器（例如 Vim）或其他编辑器，如 VSCode 或 Notepad++。
- **路径环境**：选择“从命令行和第三方软件使用 Git”以将 Git 添加到系统 PATH。
- **HTTPS 传输**：选择“使用 OpenSSL 库”（推荐）。
- **换行符转换**：选择“检出 Windows 风格，提交 Unix 风格的换行符”（默认）。

按照向导完成安装。

## 3. 验证安装
1. 打开 **Git Bash**（自动安装）。
2. 检查 Git 版本：
   ```bash
   git --version
   # 示例输出：git version 2.x.x.windows.x 
   ```

## 4. 配置 Git 用户信息

设置 Git 的用户名和邮箱：

```
bash复制git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

## 5. 配置 SSH 密钥（可选）

对于远程仓库操作（例如 GitHub、GitLab），需设置 SSH 密钥：

1. 生成 SSH 密钥：

   ```bash
   ssh-keygen -t rsa -b 4096 -C "你的邮箱@example.com"
   ```

2. 保存密钥（默认路径：`~/.ssh/id_rsa`）。

3. 将公钥（`id_rsa.pub`）添加到远程仓库：

   - 查看公钥：

     ```bash
     cat ~/.ssh/id_rsa.pub
     ```

   - 将公钥添加到你的仓库平台（例如，[GitHub SSH 设置](https://github.com/settings/keys)）。

## 6. 测试 SSH 连接

验证与 GitHub 的 SSH 连接：

```bash
ssh -T git@github.com
# 示例输出：Hi 用户名！你已成功认证...
```

# 在 Linux/Mac 上配置 Git

## 1. 生成 SSH 密钥

为 GitHub 和 GitLab 生成密钥：

```bash
# GitHub 密钥
ssh-keygen -t rsa -b 4096 -C "你的.github@邮箱.com" -f ~/.ssh/id_rsa_github

# GitLab 密钥
ssh-keygen -t ed25519 -C "你的.gitlab@邮箱.com" -f ~/.ssh/id_rsa_gitlab
```

## 2. 配置 SSH

编辑或创建 `~/.ssh/config` 文件：

```
bash复制# GitHub 配置
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa_github
    IdentitiesOnly yes

# GitLab 配置
Host gitlab.com
    HostName gitlab.com
    User git
    IdentityFile ~/.ssh/id_rsa_gitlab
    IdentitiesOnly yes
```

## 3. 将公钥添加到平台

```bash
# GitHub
cat ~/.ssh/id_rsa_github.pub

# GitLab
cat ~/.ssh/id_rsa_gitlab.pub
```

将公钥添加到相应平台（例如，[GitHub SSH 设置](https://github.com/settings/keys)）。

## 4. 测试 SSH 连接

```bash
# GitHub
ssh -T git@github.com

# GitLab
ssh -T git@gitlab.com
# 示例输出：Hi 用户名！你已成功认证...
```

## 5. 配置 Git 用户信息

全局配置（不建议用于多个账户）：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

每个仓库的配置（推荐）：

```bash
cd /path/to/repo
git config user.name "工作账户"
git config user.email "work@example.com"
```

## 6. 调试连接

```bash
ssh -vT git@github.com
```

## 7. 克隆仓库

```
bash复制# GitHub
git clone git@github.com:用户名/仓库.git

# GitLab
git clone git@gitlab.com:用户名/仓库.git
```
