# -*- coding: utf-8 -*-
import socket
from enum import Enum
from joblib import load
import paramiko
import os
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from functools import wraps
import threading
from datetime import datetime

SFTP_HOST = '10.0.37.207'
SFTP_PORT = 22
SFTP_USERNAME = 'templogs'
SFTP_PASSWORD = 'Y7Y@zabmf!SMR3'
CONN_TIMEOUT = 10  # 连接层超时
TRANSFER_TIMEOUT = 30  # 传输层超时
POOL_MAX_SIZE = 5  # 连接池容量


class Algorithm(Enum):
    MULTIPLE = "multiple"
    FOREST = "forest"
    TREE = "tree"


# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('sftp.log', maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 连接池全局变量
SSH_POOL = {}
SSH_LOCK = threading.Lock()


@contextmanager
def ssh_connection_pool():
    key = (SFTP_HOST, SFTP_PORT, SFTP_USERNAME)
    with SSH_LOCK:
        if key in SSH_POOL:
            ssh = SSH_POOL[key]
            transport = ssh.get_transport()
            if not transport or not transport.is_active():
                logger.warning("检测到失效连接，重建中...")
                del SSH_POOL[key]

        if key not in SSH_POOL:
            if len(SSH_POOL) >= POOL_MAX_SIZE:
                oldest_key = next(iter(SSH_POOL))
                logger.warning(f"连接池已满，移除旧连接: {oldest_key}")
                SSH_POOL.pop(oldest_key)

            try:
                # 创建Socket并设置超时
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(CONN_TIMEOUT)
                sock.connect((SFTP_HOST, SFTP_PORT))

                # 创建Transport并绑定Socket
                transport = paramiko.Transport(sock)
                transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)

                # 创建SSHClient并关联Transport
                ssh = paramiko.SSHClient()
                ssh._transport = transport
                SSH_POOL[key] = ssh
                logger.info(f"创建新连接: {key}")
            except (socket.error, paramiko.SSHException) as e:
                logger.error(f"连接失败: {str(e)}")
                raise

        yield SSH_POOL[key]


def log_operations(func):
    """增强型日志装饰器[3](@ref)"""

    @wraps(func)
    def wrapper(*args, ** kwargs):
        start_time = datetime.now()
        logger.info(f"操作开始: {func.__name__} | 参数: {kwargs}")
        try:
            result = func(*args, ** kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"操作成功 | 耗时: {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"操作失败 | 错误类型: {type(e).__name__} | 详情: {str(e)}")
            raise

    return wrapper


@log_operations
def load_joblib(remote_path):
    """增强版模型加载方法"""
    local_temp_path = os.path.join(os.getcwd(), 'temp.joblib')

    try:
        with ssh_connection_pool() as ssh:
            # 创建SFTP会话（无需额外设置超时）
            sftp = ssh.open_sftp()  # 继承transport层的超时设置
            logger.debug(f"SFTP会话建立 | 远程路径: {remote_path}")

            # 执行文件传输
            sftp.get(remote_path, local_temp_path)
            logger.info(f"文件下载完成 | 本地路径: {local_temp_path}")

            # 加载模型
            return load(local_temp_path)

    except paramiko.SFTPError as e:
        logger.error(f"SFTP操作失败 | 错误码: {e.code} | 详情: {str(e)}")
        raise
    finally:
        # 资源清理（增强容错）
        if 'sftp' in locals():
            try:
                sftp.close()
                logger.debug("SFTP会话已关闭")
            except Exception as e:
                logger.warning(f"SFTP关闭异常: {str(e)}")
        if os.path.exists(local_temp_path):
            try:
                os.remove(local_temp_path)
                logger.debug("临时文件已清理")
            except Exception as e:
                logger.error(f"文件删除失败: {str(e)}")

def load_model(mmsi, model_code):
    model_path = f"{get_path(mmsi, model_code)}/model.joblib"
    return load_joblib(model_path)

def load_scaler(mmsi, model_code):
    scaler_path = f"{get_path(mmsi, model_code)}/scaler.joblib"
    return load_joblib(scaler_path)

def train_data_path(mmsi, model_code):
    return f"{get_path(mmsi, model_code)}/data.csv"

def validate_data_path(mmsi, model_code):
    return f"{get_path(mmsi, model_code)}/validate.csv"

def get_path(mmsi, model_code):
    return f"/root/shipai/ftp/model/{mmsi}/{model_code}"