# LangGraph 持久化迁移方案：从 Redis 到 PostgreSQL

## 1. 背景与问题分析

### 1.1 当前问题
当前系统使用 Redis 作为 LangGraph 智能体（Agent）的状态检查点（Checkpoint）存储。随着对话轮数的增加和 Checkpoint 机制的运行，出现了以下严重问题：
- **内存溢出风险**：Redis 是内存数据库，LangGraph 保存的全量状态快照（包含完整的对话历史）会迅速消耗服务器内存。实测发现单个 Session 的 Checkpoint 甚至可达 20MB+，导致 Redis 内存占用居高不下。
- **持久化不可靠**：为了节省内存，Redis 数据往往配置为易失的或设置较短的 TTL，这与“持久化存储”的目标相悖，可能导致长周期的对话状态丢失。
- **查询能力受限**：Redis Key-Value 结构难以支持复杂的状态查询（如“查找过去 24 小时内调用过特定工具的所有对话”）。

### 1.2 解决方案：PostgreSQL
迁移至 PostgreSQL（关系型数据库）是生产环境的最佳实践：
- **磁盘存储**：数据存储在磁盘（如 `/mnt/data`），容量大且成本低，仅将活跃数据缓存在内存中。
- **JSONB 支持**：Postgres 强大的 JSONB 类型天然契合 LangGraph 的非结构化状态数据，支持高效的索引和查询。
- **官方支持**：LangGraph 官方提供了 `langgraph-checkpoint-postgres` 库，支持异步操作和连接池管理，性能优异且稳定。

---

## 2. PostgreSQL 部署与配置指南

本节说明如何在 Linux 服务器上安装 PostgreSQL，并**将数据存储路径配置为 `/mnt/data`** 以利用大容量磁盘。

### 2.1 安装 PostgreSQL
以 Ubuntu/Debian 为例：

```bash
# 1. 更新包列表
sudo apt update

# 2. 安装 PostgreSQL (通常安装最新稳定版，如 v18)
sudo apt install postgresql postgresql-contrib -y

# 3. 验证服务状态
sudo systemctl status postgresql
```

### 2.2 修改数据存储目录 (核心步骤)
默认情况下，Postgres 数据存储在 `/var/lib/postgresql`。以下步骤将其迁移至 `/mnt/data`。

#### 第一步：停止服务
在移动数据前必须停止数据库服务。
```bash
sudo systemctl stop postgresql
```

#### 第二步：迁移数据文件
假设您的挂载盘在 `/mnt/data`。

```bash
# 1. 创建新的目录结构
sudo mkdir -p /mnt/data/postgresql

# 2. 更改目录权限（必须归属 postgres 用户）
sudo chown -R postgres:postgres /mnt/data/postgresql
sudo chmod 700 /mnt/data/postgresql

# 3. 使用 rsync 同步原有数据（保留权限和属性）
# 注意：/var/lib/postgresql/X.Y/main 是默认路径，请根据实际版本号调整
# 这里以 18 为例
sudo rsync -av /var/lib/postgresql/18/main /mnt/data/postgresql/18/
```

#### 第三步：修改配置文件
编辑 `postgresql.conf` 文件。

```bash
# 查找配置文件位置
sudo find /etc/postgresql -name postgresql.conf
# 通常在 /etc/postgresql/18/main/postgresql.conf

# 编辑文件
sudo nano /etc/postgresql/18/main/postgresql.conf
```

在文件中找到 `data_directory` 配置项，修改为新路径：

```ini
# data_directory = '/var/lib/postgresql/18/main'  # 原有配置
data_directory = '/mnt/data/postgresql/18/main'   # 新配置
```

#### 第四步：重启服务并验证
```bash
# 启动服务
sudo systemctl start postgresql

# 验证数据目录是否生效
sudo -u postgres psql -c "SHOW data_directory;"
# 输出应显示：/mnt/data/postgresql/18/main
```

### 2.3 创建数据库和用户
为 LangGraph 创建专用的数据库和用户。

```bash
sudo -u postgres psql
```

在 SQL 提示符下执行：

```sql
-- 1. 创建用户
CREATE USER ai_agent WITH PASSWORD 'YourStrongPassword';

-- 2. 创建数据库
CREATE DATABASE langgraph_checkpoints OWNER ai_agent;

-- 3. 授予权限 (可选，OWNER 默认拥有权限)
GRANT ALL PRIVILEGES ON DATABASE langgraph_checkpoints TO ai_agent;

-- 退出
\q
```

---

## 3. Python 项目改造方案

### 3.1 添加依赖
在 `requirements.txt` 中添加以下依赖：

```text
langgraph-checkpoint-postgres==1.0.1
psycopg[binary,pool]>=3.1.8
```

执行安装：
```bash
pip install langgraph-checkpoint-postgres psycopg[binary,pool]
```

### 3.2 环境变量配置
在 `.env` 文件或 `core/config.py` 中添加数据库连接串配置。

**`.env` 示例:**
```ini
# Checkpoint Database
CHECKPOINT_DB_URI=postgresql://ai_agent:YourStrongPassword@localhost:5432/langgraph_checkpoints
```

**`core/config.py` 更新:**
```python
class Config:
    # ... 其他配置 ...
    CHECKPOINT_DB_URI = os.getenv("CHECKPOINT_DB_URI")
```

### 3.3 代码重构

#### A. 废弃 `core/persistence.py`
原有的 `PickleRedisSaver` 类可以废弃或删除。

#### B. 更新 `graph/main_agent.py`
使用官方的 `AsyncPostgresSaver` 替换 Redis Saver。

```python
# graph/main_agent.py

import logging
from contextlib import asynccontextmanager
# 引入 Postgres Saver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from core.config import Config

logger = logging.getLogger(__name__)

# 全局连接池管理（建议）
# 注意：实际生产中，连接池的生命周期应与应用生命周期绑定
# 这里展示最简集成方式

# 获取数据库连接串
DB_URI = Config.CHECKPOINT_DB_URI

if not DB_URI:
    logger.error("CHECKPOINT_DB_URI not set, cannot initialize PostgresSaver")
    checkpointer = None
else:
    # 初始化连接池配置
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }

    # 创建 Saver 实例的辅助函数或上下文管理器
    # 由于 workflow.compile 需要一个已初始化的 checkpointer 实例，
    # 对于异步 Saver，我们需要确保在使用前已执行 .setup()
    
    # 方案：在应用启动时初始化（如果是在 Django/ASGI 环境下）
    # 或者使用 LangGraph 的机制。
    
    # 简单实现：使用同步封装或在 ASGI 启动事件中初始化
    # 以下为适配现有代码结构的集成示例：
    
    # 注意：AsyncPostgresSaver 需要在 async 环境下运行 setup()
    # 如果 master_app 是全局编译的，这里需要特殊处理
    
    # 推荐做法：
    # 1. 创建一个全局的 pool
    pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)
    
    # 2. 实例化 Saver
    checkpointer = AsyncPostgresSaver(pool)
    
    # 注意：首次运行需要创建表。
    # 可以编写一个独立的 management command 来运行 `await checkpointer.setup()`
    # 或者在首次请求时懒加载（会有并发风险）

# 编译 Graph
# master_app = workflow.compile(checkpointer=checkpointer)
```

**关键提示：数据库表初始化**
`langgraph-checkpoint-postgres` 需要在使用前创建表结构。由于 `AsyncPostgresSaver.setup()` 是异步方法，建议创建一个 Django Management Command 来执行初始化：

**新增文件 `management/commands/init_checkpoints.py`:**
```python
from django.core.management.base import BaseCommand
import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from core.config import Config

class Command(BaseCommand):
    help = 'Initialize LangGraph checkpoint tables in PostgreSQL'

    def handle(self, *args, **options):
        uri = Config.CHECKPOINT_DB_URI
        if not uri:
            self.stdout.write(self.style.ERROR('CHECKPOINT_DB_URI is not set'))
            return

        async def _setup():
            async with AsyncConnectionPool(conninfo=uri) as pool:
                checkpointer = AsyncPostgresSaver(pool)
                await checkpointer.setup()
                self.stdout.write(self.style.SUCCESS('Successfully initialized checkpoint tables'))

        asyncio.run(_setup())
```

---

## 4. 迁移与回滚计划

### 4.1 迁移步骤
1.  **部署数据库**：按照第 2 节在服务器上安装并配置 PostgreSQL，确保数据目录在 `/mnt/data`。
2.  **更新代码**：修改 `requirements.txt` 和 `config.py`，实现 `AsyncPostgresSaver` 逻辑。
3.  **初始化表**：运行 `python manage.py init_checkpoints` 创建表结构。
4.  **发布上线**：重启 Django 服务。
5.  **清理旧数据**：确认系统运行稳定后，清理 Redis 中的 `checkpoint:*` 键。

### 4.2 注意事项
- **旧数据兼容性**：此方案**不会**自动将 Redis 中的旧 Checkpoint 迁移到 Postgres。切换后，用户之前的对话上下文将丢失（相当于开启新会话）。鉴于 Redis 中数据已有问题，直接丢弃旧状态通常是可接受的。
- **连接池调优**：根据并发量调整 `max_size` 参数，避免数据库连接耗尽。
