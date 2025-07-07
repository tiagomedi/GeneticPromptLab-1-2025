#!/usr/bin/env python3
"""
Test script for SSH Connection Manager
Demonstrates usage of the generic SSH connection system
"""

import sys
import time
from ssh_connection import SSHConnectionManager, LLMRemoteExecutor

def test_basic_connection():
    """Test basic SSH connection functionality"""
    print("🧪 Testing basic SSH connection...")
    
    ssh_manager = SSHConnectionManager()
    
    try:
        if ssh_manager.connect():
            print("✅ SSH connection successful")
            
            # Test basic command
            success, stdout, stderr = ssh_manager.execute_command("whoami")
            if success:
                print(f"✅ Remote user: {stdout.strip()}")
            else:
                print(f"❌ Command failed: {stderr}")
            
            ssh_manager.close()
            return True
        else:
            print("❌ SSH connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_llm_availability():
    """Test LLM availability and setup"""
    print("\n🧪 Testing LLM availability...")
    
    ssh_manager = SSHConnectionManager()
    
    try:
        with ssh_manager.connection_context():
            if ssh_manager.test_llm_availability():
                print("✅ LLM is available and working")
                return True
            else:
                print("❌ LLM is not available")
                return False
                
    except Exception as e:
        print(f"❌ LLM availability test failed: {e}")
        return False

def test_simple_prompt():
    """Test executing a simple prompt"""
    print("\n🧪 Testing simple prompt execution...")
    
    ssh_manager = SSHConnectionManager()
    
    try:
        with ssh_manager.connection_context():
            prompt = "What is machine learning? Answer in one sentence."
            success, response = ssh_manager.execute_llm_prompt(prompt)
            
            if success:
                print(f"✅ Prompt executed successfully")
                print(f"📝 Response: {response}")
                return True
            else:
                print(f"❌ Prompt execution failed: {response}")
                return False
                
    except Exception as e:
        print(f"❌ Prompt test failed: {e}")
        return False

def test_high_level_interface():
    """Test the high-level LLMRemoteExecutor interface"""
    print("\n🧪 Testing high-level interface...")
    
    executor = LLMRemoteExecutor()
    
    try:
        # Test setup
        if executor.test_setup():
            print("✅ High-level setup test passed")
            
            # Test single prompt
            success, response = executor.execute_prompt(
                "Explain genetic algorithms in exactly 10 words."
            )
            
            if success:
                print(f"✅ Single prompt test passed")
                print(f"📝 Response: {response}")
                
                # Test batch prompts
                prompts = [
                    "What is AI?",
                    "What is machine learning?",
                    "What is deep learning?"
                ]
                
                print("\n🔄 Testing batch prompts...")
                results = executor.execute_batch_prompts(prompts)
                
                successful_results = [r for r in results if r[0]]
                print(f"✅ Batch test: {len(successful_results)}/{len(prompts)} prompts successful")
                
                for i, (success, response) in enumerate(results):
                    if success:
                        print(f"   📝 Prompt {i+1}: {response[:100]}...")
                    else:
                        print(f"   ❌ Prompt {i+1} failed: {response}")
                
                return len(successful_results) > 0
            else:
                print(f"❌ Single prompt test failed: {response}")
                return False
        else:
            print("❌ High-level setup test failed")
            return False
            
    except Exception as e:
        print(f"❌ High-level interface test failed: {e}")
        return False

def test_interactive_commands():
    """Test interactive command execution"""
    print("\n🧪 Testing interactive commands...")
    
    ssh_manager = SSHConnectionManager()
    
    try:
        with ssh_manager.connection_context():
            # Test simple interactive command
            success, output = ssh_manager.execute_interactive_command(
                "echo 'Interactive test'",
                inputs=["echo 'Second command'"],
                read_delay=1.0,
                timeout=10
            )
            
            if success and "Interactive test" in output:
                print("✅ Interactive command test passed")
                print(f"📝 Output preview: {output[:100]}...")
                return True
            else:
                print(f"❌ Interactive command test failed")
                return False
                
    except Exception as e:
        print(f"❌ Interactive command test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 SSH Connection Manager - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Connection", test_basic_connection),
        ("LLM Availability", test_llm_availability),
        ("Simple Prompt", test_simple_prompt),
        ("Interactive Commands", test_interactive_commands),
        ("High-Level Interface", test_high_level_interface)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            results[test_name] = (result, duration)
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status} - {test_name} ({duration:.1f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            results[test_name] = (False, duration)
            print(f"\n❌ FAILED - {test_name} ({duration:.1f}s): {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary:")
    print(f"{'='*60}")
    
    passed = sum(1 for result, _ in results.values() if result)
    total = len(results)
    
    for test_name, (result, duration) in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name:<25} ({duration:.1f}s)")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SSH connection system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the configuration and network connectivity.")
        return False

def main():
    """Main test function"""
    try:
        success = run_comprehensive_test()
        return success
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 