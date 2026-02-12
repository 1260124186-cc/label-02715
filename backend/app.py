"""
PPO强化学习算法 - Flask Web API
提供HTTP接口来触发训练和查看结果
"""

from flask import Flask, jsonify, send_file, request
import os
import json
import threading
from datetime import datetime

from ppo import train_ppo

app = Flask(__name__)

# 全局变量存储训练状态
training_status = {
    'is_training': False,
    'progress': 0,
    'message': '空闲',
    'last_result': None
}

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/app/output')


@app.route('/')
def index():
    """首页 - API说明"""
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
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/status')
def status():
    """查看训练状态"""
    return jsonify(training_status)


@app.route('/train', methods=['POST'])
def train():
    """开始训练"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({
            'success': False,
            'message': '训练正在进行中，请等待完成'
        }), 400
    
    # 获取训练参数
    data = request.get_json() or {}
    max_episodes = data.get('max_episodes', 300)
    
    def run_training():
        global training_status
        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['message'] = '训练中...'
        
        try:
            import torch
            import numpy as np
            
            # 设置随机种子
            torch.manual_seed(42)
            np.random.seed(42)
            
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
            
        except Exception as e:
            training_status['message'] = f'训练失败: {str(e)}'
            
        finally:
            training_status['is_training'] = False
            training_status['progress'] = 100
    
    # 在后台线程中运行训练
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': '训练已开始',
        'max_episodes': max_episodes
    })


@app.route('/results')
def results():
    """获取训练结果"""
    results_path = os.path.join(OUTPUT_DIR, 'training_results.json')
    
    if not os.path.exists(results_path):
        return jsonify({
            'success': False,
            'message': '暂无训练结果，请先进行训练'
        }), 404
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return jsonify({
        'success': True,
        'data': data
    })


@app.route('/plot')
def plot():
    """获取训练曲线图"""
    plot_path = os.path.join(OUTPUT_DIR, 'ppo_training_results.png')
    
    if not os.path.exists(plot_path):
        return jsonify({
            'success': False,
            'message': '暂无训练曲线图，请先进行训练'
        }), 404
    
    return send_file(plot_path, mimetype='image/png')


if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)
