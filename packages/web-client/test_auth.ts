#!/usr/bin/env bun

/**
 * 测试认证系统错误处理
 */

import { useAuth } from './src/auth';

// 模拟localStorage
const localStorageMock = {
  getItem: (key: string) => null,
  setItem: (key: string, value: string) => {},
  removeItem: (key: string) => {},
};

// 在Node.js环境中模拟
global.localStorage = localStorageMock as any;

// 模拟axios
const mockAxios = {
  get: () => Promise.reject(new Error('Network Error')),
  interceptors: {
    request: { use: () => {} }
  },
  defaults: {}
};

(global as any).axios = mockAxios;

// 测试认证系统
async function testAuthSystem() {
  console.log('🧪 测试认证系统错误处理...');

  try {
    const auth = useAuth();

    // 测试初始状态
    console.log('初始状态:', auth.status());

    // 测试登出
    console.log('执行登出...');
    await auth.logout();

    // 测试登出后的状态
    console.log('登出后状态:', auth.status());

    console.log('✅ 认证系统错误处理测试完成');

  } catch (error) {
    console.error('❌ 测试失败:', error);
  }
}

// 运行测试
if (import.meta.main) {
  testAuthSystem();
}

export { testAuthSystem };