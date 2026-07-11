# Git 同步 SOP

最后更新：2026-07-11

## 仓库信息

| 项目 | 值 |
|------|-----|
| GitHub 地址 | `https://github.com/DuDouVagrant/DFS_HK.git` |
| 工作分支 | `YangmingLi-CUHK-patch-1` |
| 本地目录名 | `DFS_HK5` |
| 大小 | ~500MB（含历史权重和数据，浅克隆大幅加速） |

---

## 一、新电脑初始化（从来没有过这个文件夹）

适用于：换新电脑、跳板机首次下载。

```bash
# 0. （只需一次）连接不稳时调大缓冲区
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300

# 1. 浅克隆（只拉最新，不拉历史）
git clone --depth 1 --branch YangmingLi-CUHK-patch-1 \
  https://github.com/DuDouVagrant/DFS_HK.git DFS_HK5

cd DFS_HK5
```

---

## 二、已有文件夹，但从未接过 Git

适用于：文件夹是你手动拷贝过来的，里面有文件但没有 `.git`。

**核心思路**：先 `init` + `remote add` + `fetch`，然后用 `reset --mixed` 告诉 git "云端最新版就是基准"，让 git 把本地和云端的差异算出来。

```bash
cd 你的/DFS_HK5/

# 1. 初始化并接上远程
git init
git remote add origin https://github.com/DuDouVagrant/DFS_HK.git
git fetch origin YangmingLi-CUHK-patch-1

# 2. 把 git 基准设成远程最新（不动你磁盘上的文件！）
git reset --mixed origin/YangmingLi-CUHK-patch-1

# 3. 设分支名和跟踪
git branch -M YangmingLi-CUHK-patch-1
git branch --set-upstream-to=origin/YangmingLi-CUHK-patch-1

# 4. 看差异（此时 git status 告诉你本地相对云端多了/少/改了啥）
git status
```

**⚠️ reset --mixed 是最关键的一步**。跳过它直接 add → commit → push，会造出一个和云端毫无关联的独立历史，push 被拒。

---

## 三、把本地改动推上云端（日常同步，本机 → GitHub）

```bash
cd 你的/DFS_HK5/

# 1. 看清楚改了这些
git status

# 2. 暂存所有改动
git add -A

# 3. 复查一遍——特别留意有没有不需要的 deleted:
git status

# 4. 提交
git commit -m "描述这次的改动"

# 5. 先 pull 再 push（避免别人在你之前推了东西导致被拒）
git pull origin YangmingLi-CUHK-patch-1

# 6. 推
git push origin YangmingLi-CUHK-patch-1
```

---

## 四、把云端改动同步下来（GitHub → 本机）

**前提：本机已经有 git 仓库（即做过第一步或第二步）。**
**警告：这会丢弃本机所有未提交的改动！**

```bash
cd 你的/DFS_HK5/

git fetch origin YangmingLi-CUHK-patch-1
git reset --hard origin/YangmingLi-CUHK-patch-1

# 可选：清掉未被跟踪的垃圾文件（让目录和远程完全一致）
git clean -fd
```

如果只想拉更新但**保留本地未提交的改动**（不覆盖）：

```bash
git fetch origin YangmingLi-CUHK-patch-1
git merge origin/YangmingLi-CUHK-patch-1
```

## 五、常见的坑

### 1. 忘了 `git add` 就直接 commit

症状：`no changes added to commit`。改动在工作区但没进暂存区。
解决：先 `git add -A` 再 `git commit`。

### 2. `git status` 里有不想推的 deleted

症状：这台电脑本地缺某些文件（比如 CLAUDE.md、某个 .pth），`git add -A` 会当成"删除"推上去。
解决：
- 如果那些文件应该保留 → `git restore 文件路径` 恢复，再 `git add -A`
- 如果确实想从云端删掉 → 直接用 `git add -A`，commit message 里说明删了什么

### 3. push 被拒 `non-fast-forward`

症状：云端有你不了解的新提交。
解决：先 `git pull origin YangmingLi-CUHK-patch-1`（合并），再 `git push`。如果有冲突就解决冲突再 commit 再 push。

### 4. `git pull` 出 CONFLICT

症状：同一个文件两边都改了。
解决：打开冲突文件，找到 `<<<<<<<` / `=======` / `>>>>>>>` 标记，手动修好 → `git add 该文件` → `git commit` → `git push`。

### 5. 新日志/权重被 `.gitignore` 挡住，push 不上去

症状：`git status` 看不到你新建的 `logs/combo_filters/` 目录。
解决：`.gitignore` 里写了 `logs/`，只拦**没被 git 跟踪过的新文件**。要推的话：

```bash
git add -f logs/combo_filters
# 注意：加一次就够了，之后就进入跟踪列表，不再被 ignore
```

已在跟踪列表里的旧文件（比如 `logs/RobustPrompt-I/`、`pre_trained_model_raw/` 下的权重）不受 `.gitignore` 影响，照常推送。

### 6. CRLF 换行符警告

症状：`warning: LF will be replaced by CRLF`。
说明：Windows 和 Linux 换行符差异，**不影响功能，忽略即可**。

### 7. 连接不稳 push/fetch 到一半断

```
error: RPC failed; curl 56 Recv failure
fatal: early EOF
```

解决：

```bash
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300
# 然后重试
```

### 8. 有多台电脑在同步，不知道哪台是最新的

原则：**GitHub 是"真相来源"**。任何一台电脑的操作前，先 `git fetch` + 看 `git log --oneline -3` 确认远程最新提交。谁先 push 谁说了算，后 push 的人用 `git pull` 合并。

---

## 六、当前仓库状态备忘（2026-07-11）

- **分支**：`YangmingLi-CUHK-patch-1`（不是 main）
- **CLAUDE.md**：已从云端删除（办公室电脑选择不保留），仅存于本机 `80406` 上
- **`.claude/settings.local.json`**：已从云端删除（本地配置文件，不影响他人）
- **被 `.gitignore` 排除的**：新 `logs/` 子目录、`*.pth`/`*.pt`/`*.npy`（但不影响已跟踪的老文件）
- **已跟踪的老二进制**（云端有）：294 个 `.pth` 权重、71 个 `.pt` 数据、84 个日志 — 这些不受 `.gitignore` 影响
- **主动推送的中间结果**：目前不推，等有明确需要时用 `git add -f` 加

---

## 七、网络连接优化（持久配置）

以下配置只需跑一次，会写入 git 全局配置（`~/.gitconfig`），对所有仓库生效：

```bash
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300
```
