"""
PPO强化学习算法 - Flask Web API
提供HTTP接口来触发训练和查看结果
"""

from flask import Flask, jsonify, send_file, request
import os
import json
import logging
import threading
import traceback
from datetime import datetime
from functools import wraps

from ppo import train_ppo

# ==================== 日志配置 ====================

def setup_logging():
    """配置结构化日志"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
        ]
    )
    
    # 创建应用日志器
    logger = logging.getLogger('ppo-api')
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

# ==================== Flask应用 ====================

app = Flask(__name__)

# 全局变量存储训练状态
training_status = {
    'is_training': False,
    'progress': 0,
    'message': '空闲',
    'last_result': None,
    'error': None
}

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')


# ==================== 错误处理装饰器 ====================

def handle_exceptions(f):
    """统一异常处理装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"文件未找到: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'file_not_found',
                'message': f'文件未找到: {str(e)}'
            }), 404
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'json_decode_error',
                'message': f'JSON解析错误: {str(e)}'
            }), 400
        except ValueError as e:
            logger.error(f"参数错误: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'value_error',
                'message': f'参数错误: {str(e)}'
            }), 400
        except Exception as e:
            logger.error(f"服务器内部错误: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': 'internal_error',
                'message': f'服务器内部错误: {str(e)}'
            }), 500
    return decorated_function


# ==================== 全局错误处理 ====================

@app.errorhandler(404)
def not_found_error(error):
    """处理404错误"""
    logger.warning(f"404错误: {request.url}")
    return jsonify({
        'success': False,
        'error': 'not_found',
        'message': '请求的资源不存在'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    logger.error(f"500错误: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'internal_error',
        'message': '服务器内部错误'
    }), 500


@app.errorhandler(405)
def method_not_allowed(error):
    """处理405错误"""
    logger.warning(f"405错误: {request.method} {request.url}")
    return jsonify({
        'success': False,
        'error': 'method_not_allowed',
        'message': f'不支持的请求方法: {request.method}'
    }), 405


# ==================== API路由 ====================

@app.route('/')
@handle_exceptions
def index():
    """首页 - API说明"""
    logger.info("访问API首页")
    return jsonify({
        'name': 'PPO强化学习算法API',
        'version': '1.0.0',
        'description': 'PPO (Proximal Policy Optimization) 强化学习算法实现',
        'endpoints': {
            '/': 'API说明',
            '/health': '健康检查',
            '/status': '查看训练状态',
            '/train': '开始训练 (POST)',
            '/results': '获取训练结果',
            '/plot': '获取训练曲线图'
        },
        'environment': 'CartPole-v1',
        'algorithm': 'PPO (Proximal Policy Optimization)'
    })


@app.route('/health')
@handle_exceptions
def health():
    """健康检查"""
    logger.debug("健康检查请求")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ppo-backend'
    })


@app.route('/status')
@handle_exceptions
def status():
    """查看训练状态"""
    logger.info(f"查询训练状态: is_training={training_status['is_training']}")
    return jsonify(training_status)


@app.route('/train', methods=['POST'])
@handle_exceptions
def train():
    """开始训练"""
    global training_status
    
    if training_status['is_training']:
        logger.warning("尝试启动训练但已有训练在进行中")
        return jsonify({
            'success': False,
            'error': 'training_in_progress',
            'message': '训练正在进行中，请等待完成'
        }), 400
    
    # 获取并验证训练参数
    data = request.get_json() or {}
    max_episodes = data.get('max_episodes', 300)
    
    # 参数验证
    if not isinstance(max_episodes, int) or max_episodes < 1:
        logger.error(f"无效的max_episodes参数: {max_episodes}")
        return jsonify({
            'success': False,
            'error': 'invalid_parameter',
            'message': 'max_episodes必须是正整数'
        }), 400
    
    if max_episodes > 10000:
        logger.warning(f"max_episodes过大: {max_episodes}，限制为10000")
        max_episodes = 10000
    
    logger.info(f"开始训练，max_episodes={max_episodes}")
    
    def run_training():
        global training_status
        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['message'] = '训练中...'
        training_status['error'] = None
        
        try:
            import torch
            import numpy as np
            
            logger.info("初始化训练环境...")
            
            # 设置随机种子
            torch.manual_seed(42)
            np.random.seed(42)
            
            logger.info(f"开始PPO训练，目标回合数: {max_episodes}")
            
            # 开始训练
            agent, episode_rewards, avg_rewards, results = train_ppo(
                env_name='CartPole-v1',
                max_episodes=max_episodes,
                max_steps=500,
                update_interval=2048,
                print_interval=10,
                output_dir=OUTPUT_DIR
            )
            
            training_status['last_result'] = results
            training_status['message'] = f'训练完成! 最终平均奖励: {results["final_avg_reward"]:.1f}'
            training_status['error'] = None
            
            logger.info(f"训练完成! 最终平均奖励: {results['final_avg_reward']:.1f}, "
                       f"是否解决: {results['solved']}")
            
        except ImportError as e:
            error_msg = f'依赖导入失败: {str(e)}'
            training_status['message'] = error_msg
            training_status['error'] = 'import_error'
            logger.error(f"训练失败 - {error_msg}\n{traceback.format_exc()}")
            
        except RuntimeError as e:
            error_msg = f'运行时错误: {str(e)}'
            training_status['message'] = error_msg
            training_status['error'] = 'runtime_error'
            logger.error(f"训练失败 - {error_msg}\n{traceback.format_exc()}")
            
        except Exception as e:
            error_msg = f'训练失败: {str(e)}'
            training_status['message'] = error_msg
            training_status['error'] = 'unknown_error'
            logger.error(f"训练失败 - {error_msg}\n{traceback.format_exc()}")
            
        finally:
            training_status['is_training'] = False
            training_status['progress'] = 100
            logger.info("训练线程结束")
    
    # 在后台线程中运行训练
    thread = threading.Thread(target=run_training, name='PPO-Training-Thread')
    thread.daemon = True
    thread.start()
    
    logger.info(f"训练线程已启动: {thread.name}")
    
    return jsonify({
        'success': True,
        'message': '训练已开始',
        'max_episodes': max_episodes
    })


@app.route('/results')
@handle_exceptions
def results():
    """获取训练结果"""
    results_path = os.path.join(OUTPUT_DIR, 'training_results.json')
    
    logger.info(f"请求训练结果，文件路径: {results_path}")
    
    if not os.path.exists(results_path):
        logger.warning("训练结果文件不存在")
        return jsonify({
            'success': False,
            'error': 'no_results',
            'message': '暂无训练结果，请先进行训练'
        }), 404
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"返回训练结果: final_avg_reward={data.get('final_avg_reward', 'N/A')}")
    
    return jsonify({
        'success': True,
        'data': data
    })


@app.route('/plot')
@handle_exceptions
def plot():
    """获取训练曲线图"""
    plot_path = os.path.join(OUTPUT_DIR, 'ppo_training_results.png')
    
    logger.info(f"请求训练曲线图，文件路径: {plot_path}")
    
    if not os.path.exists(plot_path):
        logger.warning("训练曲线图文件不存在")
        return jsonify({
            'success': False,
            'error': 'no_plot',
            'message': '暂无训练曲线图，请先进行训练'
        }), 404
    
    logger.info("返回训练曲线图")
    return send_file(plot_path, mimetype='image/png')


# ==================== 启动入口 ====================

if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    # 获取端口（默认5001，避免与macOS AirPlay冲突）
    port = int(os.environ.get('PORT', 5001))
    
    # 启动Flask应用
    logger.info("启动PPO强化学习API服务...")
    logger.info(f"服务地址: http://0.0.0.0:{port}/")
    app.run(host='0.0.0.0', port=port, debug=False)
