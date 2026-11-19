# 启动DSH
npx @deepseek-ai/dsh@latest web

# 虚拟环境
- 名字：dryer-ts
- 位置：/opt/conda_env/dryer-ts
- 
```bash
conda active dryer-ts
```


# 上传至GITHUB
```bash
**首次提交代码**
cd ~/hdt_dryer_mpc
# 将当前项目的所有核心文件添加到暂存区
git add .
# 仅取消暂存，保留本地代码修改（最常用）
git reset
# 检查暂存区状态，确认没有多余的临时或结果大文件
git status
# 提交到本地版本库
git commit -m "feat: initial commit"
# 将当前分支重命名为 main（与远程默认分支对齐）
git branch -M main
# 关联你的 GitHub 远程仓库
https://github.com/DontLookBackIn-Anger/hdt_dryer_mpc
git remote add origin https://github.com/DontLookBackIn-Anger/hdt_dryer_mpc.git
# 验证是否成功
git remote -v
# 推送至远程仓库并建立上游追踪
git push -u origin main

**后续提交代码**
cd samples/tsn-distributed-control
git add .
git commit -m "aa"
git push -u origin main
```