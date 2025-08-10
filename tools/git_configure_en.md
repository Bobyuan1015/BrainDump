# Configuring Git on Windows

## 1. Install Git
1. Visit the [Git official website](https://git-scm.com/).
2. Download the Windows version of Git.
3. Run the installer after downloading.

## 2. Configure Installation Options
During installation, Git prompts for configuration options:
- **Installation Directory**: Use the default path or choose a custom one.
- **Editor**: Select the default editor (e.g., Vim) or another like VSCode or Notepad++.
- **Path Environment**: Choose "Git from the command line and 3rd-party software" to add Git to system PATH.
- **HTTPS Transport**: Select "Use the OpenSSL library" (recommended).
- **Line Ending Conversion**: Choose "Checkout Windows-style, commit Unix-style line endings" (default).

Complete the installation by following the wizard.

## 3. Verify Installation
1. Open **Git Bash** (installed automatically).
2. Check the Git version:
   ```bash
   git --version
   # Example output: git version 2.x.x.windows.x 
   ```
   
## 4. Configure Git User Information

Set your username and email for Git:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 5. Configure SSH Key (Optional)

For remote repository operations (e.g., GitHub, GitLab), set up an SSH key:

1. Generate an SSH key:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
   ```

2. Save the key (default: `~/.ssh/id_rsa`).

3. Add the public key (`id_rsa.pub`) to the remote repository:

   - View the public key:

     ```bash
     cat ~/.ssh/id_rsa.pub
     ```

   - Add it to your repository platform (e.g., [GitHub SSH settings](https://github.com/settings/keys)).

## 6. Test SSH Connection 

Verify the SSH connection to GitHub:

```bash
ssh -T git@github.com
# Example output: Hi username! You've successfully authenticated...
```



# Configuring Git on Linux/Mac

## 1. Generating SSH Keys

Generate keys for GitHub and GitLab:

```bash
# GitHub key
ssh-keygen -t rsa -b 4096 -C "your.github@email.com" -f ~/.ssh/id_rsa_github

# GitLab key
ssh-keygen -t ed25519 -C "your.gitlab@email.com" -f ~/.ssh/id_rsa_gitlab
```

## 2. Configuring SSH
Edit or create the ~/.ssh/config file:

```bash
# GitHub configuration
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa_github
    IdentitiesOnly yes

# GitLab configuration
Host gitlab.com
    HostName gitlab.com
    User git
    IdentityFile ~/.ssh/id_rsa_gitlab
    IdentitiesOnly yes
```

## 3. Adding Public Keys to Platforms
```bash
# GitHub
cat ~/.ssh/id_rsa_github.pub

# GitLab
cat ~/.ssh/id_rsa_gitlab.pub
```
Add the public key to the respective platform (e.g.,  [GitHub SSH settings](https://github.com/settings/keys)).

## 4. Testing SSH Connection
```bash
# GitHub
ssh -T git@github.com

# GitLab
ssh -T git@gitlab.com
# Example output: Hi username! You've successfully authenticated...
```
## 5. Configuring Git User Information
Global configuration (not recommended for multiple accounts):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
Per-repository configuration (recommended):
```
```bash
cd /path/to/repo
git config user.name "Work Account"
git config user.email "work@example.com"
```

## 6. Debugging Connection
```bash
ssh -vT git@github.com
```
## 7. Cloning Repositories
```bash
# GitHub
git clone git@github.com:username/repo.git

# GitLab
git clone git@gitlab.com:username/repo.git
```