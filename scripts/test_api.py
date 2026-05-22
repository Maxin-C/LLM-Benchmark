#!/usr/bin/env python3
"""
API连接测试脚本 - 验证API Key和模型可用性
"""

import os
from openai import OpenAI

def test_api_connection(api_key: str, base_url: str, model: str, params: dict = None) -> dict:
    """
    测试单个API连接
    
    参数：
        api_key: API密钥
        base_url: API基础URL
        model: 模型名称
        params: 额外参数
    
    返回：
        测试结果字典
    """
    result = {
        'api_key': api_key[:10] + '...' + api_key[-10:],  # 隐藏中间部分
        'base_url': base_url,
        'model': model,
        'success': False,
        'error': None,
        'response': None,
        'latency': None
    }
    
    if not params:
        params = {
            'temperature': 0.1,
            'top_p': 0.9,
            'max_tokens': 50
        }
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        import time
        start_time = time.time()
        
        chat_completion = client.chat.completions.create(
            messages=[{'role': 'user', 'content': 'Hello, test connection.'}],
            model=model,
            **params
        )
        
        latency = time.time() - start_time
        
        result['success'] = True
        result['response'] = chat_completion.choices[0].message.content.strip()
        result['latency'] = round(latency, 2)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    print("="*70)
    print("              API连接测试脚本")
    print("="*70)
    
    # 加载.env文件
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ 成功加载.env文件")
    except ImportError:
        print("⚠️ 未安装python-dotenv，尝试直接读取环境变量")
    
    # 从环境变量读取配置
    api_key = os.getenv('EASE_LLM_API_KEY', '')
    base_url = os.getenv('EASE_LLM_BASE_URL', 'https://api.pumpkinaigc.online/v1')
    
    print(f"\n测试配置：")
    print(f"  API Key: {api_key[:10]}...{api_key[-10:]}")
    print(f"  Base URL: {base_url}")
    print()
    
    # 测试的模型列表（第三方API支持）
    models_to_test = [
        {'name': 'gpt-4o', 'description': '待评估模型 (虚拟患者)'},
        {'name': 'deepseek-v3', 'description': '虚拟医生/评估器/监控器'},
    ]
    
    print("测试结果：")
    print("-"*70)
    
    all_passed = True
    results = []
    
    for model_info in models_to_test:
        print(f"\n测试模型: {model_info['name']}")
        print(f"描述: {model_info['description']}")
        
        result = test_api_connection(api_key, base_url, model_info['name'])
        
        if result['success']:
            print(f"✅ 成功")
            print(f"   响应: {result['response'][:50]}...")
            print(f"   延迟: {result['latency']}秒")
        else:
            print(f"❌ 失败")
            print(f"   错误: {result['error']}")
            all_passed = False
        
        results.append(result)
    
    print("\n" + "="*70)
    
    # 总结报告
    passed_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"测试总结: {passed_count}/{total_count} 模型测试通过")
    
    if all_passed:
        print("🎉 所有测试通过！API Key有效，模型可用。")
    else:
        print("⚠️ 部分测试失败，请检查API Key或联系服务提供商。")
    
    # 诊断建议
    print("\n诊断建议：")
    for i, result in enumerate(results):
        if not result['success']:
            print(f"\n模型 {result['model']} 失败:")
            if '400' in result['error']:
                print("  - 可能原因: 参数错误或模型不支持")
                print("  - 建议: 检查top_p参数是否在0-1范围内")
            elif '401' in result['error'] or '403' in result['error']:
                print("  - 可能原因: API Key无效或过期")
                print("  - 建议: 检查API Key是否正确")
            elif 'timeout' in result['error'].lower():
                print("  - 可能原因: 网络超时")
                print("  - 建议: 检查网络连接或稍后重试")
            else:
                print(f"  - 错误详情: {result['error']}")

if __name__ == '__main__':
    main()