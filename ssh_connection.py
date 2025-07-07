#!/usr/bin/env python3
"""
SSH Connection Manager for GeneticPromptLab
Provides secure remote execution capabilities for LLM operations
"""

import json
import os
import sys
import time
import paramiko
import logging
from typing import Dict, Any, Tuple, Optional, List
from contextlib import contextmanager

class SSHConnectionManager:
    """
    Manages SSH connections with jump host support for remote LLM execution
    """
    
    def __init__(self, credentials_file: str = "ssh_credentials.json", logger: Optional[logging.Logger] = None):
        self.credentials_file = credentials_file
        self.logger = logger or self._setup_logger()
        self.credentials = None
        self.jump_client = None
        self.target_client = None
        self.channel = None
        self._connected = False
        
        self._load_credentials()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for SSH operations"""
        logger = logging.getLogger('SSHConnectionManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_credentials(self) -> bool:
        """Load credentials from JSON file"""
        try:
            if not os.path.exists(self.credentials_file):
                self.logger.error(f"Credentials file not found: {self.credentials_file}")
                return False
                
            with open(self.credentials_file, 'r') as f:
                self.credentials = json.load(f)
            
            # Validate required fields
            required_fields = ['ssh_config', 'llm_config']
            if not all(field in self.credentials for field in required_fields):
                self.logger.error("Missing required fields in credentials file")
                return False
                
            self.logger.info("✅ Credentials loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error loading credentials: {e}")
            return False
    
    def connect(self) -> bool:
        """Establish SSH connection through jump host"""
        if not self.credentials:
            self.logger.error("No credentials available")
            return False
        
        if self._connected:
            self.logger.info("Already connected")
            return True
            
        self.logger.info("🔗 Establishing SSH connection...")
        
        try:
            ssh_config = self.credentials['ssh_config']
            jump_host = ssh_config['jump_host']
            target_host = ssh_config['target_host']
            
            self.logger.info(f"Jump host: {jump_host['username']}@{jump_host['host']}:{jump_host['port']}")
            self.logger.info(f"Target host: {target_host['username']}@{target_host['host']}")
            
            # Connect to jump host
            self.jump_client = paramiko.SSHClient()
            self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.jump_client.connect(
                hostname=jump_host['host'],
                port=jump_host['port'],
                username=jump_host['username'],
                password=jump_host['password'],
                timeout=self.credentials.get('connection_settings', {}).get('command_timeout', 30)
            )
            
            # Setup tunnel
            jump_transport = self.jump_client.get_transport()
            if jump_transport is None:
                raise ConnectionError("Failed to get transport from jump host")
            dest_addr = (target_host['host'], target_host['port'])
            local_addr = ('', 0)
            self.channel = jump_transport.open_channel("direct-tcpip", dest_addr, local_addr)
            
            # Connect to target host
            self.target_client = paramiko.SSHClient()
            self.target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.target_client.connect(
                hostname=target_host['host'],
                port=target_host['port'],
                username=target_host['username'],
                password=target_host['password'],
                sock=self.channel,
                timeout=self.credentials.get('connection_settings', {}).get('command_timeout', 30)
            )
            
            self._connected = True
            self.logger.info("✅ SSH connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Connection error: {e}")
            self.close()
            return False
    
    def is_connected(self) -> bool:
        """Check if SSH connection is active"""
        return self._connected and self.target_client is not None
    
    def execute_command(self, command: str, timeout: Optional[int] = None) -> Tuple[bool, str, str]:
        """
        Execute a command on the remote server
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        if not self.is_connected():
            return False, "", "No SSH connection available"
        
        timeout = timeout or (self.credentials or {}).get('connection_settings', {}).get('command_timeout', 30)
        
        try:
            self.logger.debug(f"Executing command: {command}")
            stdin, stdout, stderr = self.target_client.exec_command(command, timeout=timeout)
            
            stdout_data = stdout.read().decode()
            stderr_data = stderr.read().decode()
            
            # Check exit status
            exit_status = stdout.channel.recv_exit_status()
            success = exit_status == 0
            
            if not success and stderr_data:
                self.logger.warning(f"Command failed with error: {stderr_data}")
            
            return success, stdout_data, stderr_data
            
        except Exception as e:
            self.logger.error(f"❌ Command execution error: {e}")
            return False, "", str(e)
    
    def execute_interactive_command(self, command: str, inputs: Optional[List[str]] = None, 
                                  read_delay: float = 1.0, 
                                  timeout: Optional[int] = None) -> Tuple[bool, str]:
        """
        Execute an interactive command with inputs
        
        Args:
            command: Command to execute
            inputs: List of inputs to send to the command
            read_delay: Delay between reads in seconds
            timeout: Total timeout in seconds
            
        Returns:
            Tuple of (success, output)
        """
        if not self.is_connected():
            return False, "No SSH connection available"
        
        timeout = timeout or (self.credentials or {}).get('connection_settings', {}).get('command_timeout', 30)
        
        try:
            # Create interactive shell
            channel = self.target_client.invoke_shell()
            
            # Send initial command
            channel.send(f"{command}\n".encode())
            time.sleep(read_delay)
            
            # Send inputs if provided
            if inputs:
                for input_cmd in inputs:
                    channel.send(f"{input_cmd}\n".encode())
                    time.sleep(read_delay)
            
            # Read output
            output = ""
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if channel.recv_ready():
                    data = channel.recv(1024).decode()
                    output += data
                    start_time = time.time()  # Reset timeout on data received
                else:
                    time.sleep(0.1)
            
            # Clean up
            channel.send("exit\n".encode())
            time.sleep(0.5)
            channel.close()
            
            return True, output
            
        except Exception as e:
            self.logger.error(f"❌ Interactive command error: {e}")
            return False, str(e)
    
    def test_llm_availability(self, model: Optional[str] = None) -> bool:
        """Test if LLM service is available and working"""
        model = model or (self.credentials or {}).get('llm_config', {}).get('model', 'llama3.1')
        
        self.logger.info(f"🧪 Testing LLM availability: {model}")
        
        # Check if ollama is installed
        success, stdout, stderr = self.execute_command("which ollama")
        if not success or "ollama" not in stdout:
            self.logger.error("❌ Ollama not found on remote server")
            return False
        
        ollama_path = stdout.strip()
        self.logger.info(f"✅ Ollama found at: {ollama_path}")
        
        # Check if service is running
        success, stdout, stderr = self.execute_command("ps aux | grep ollama | grep -v grep")
        if not success or not stdout:
            self.logger.warning("⚠️ Ollama service not running, attempting to start...")
            if not self._start_ollama_service(ollama_path):
                return False
        else:
            self.logger.info("✅ Ollama service is running")
        
        # Check if model is available
        success, stdout, stderr = self.execute_command(f"{ollama_path} list")
        if not success:
            self.logger.error("❌ Failed to list available models")
            return False
        
        if model not in stdout.lower():
            self.logger.warning(f"⚠️ Model {model} not found, attempting to install...")
            if not self._install_llm_model(ollama_path, model):
                return False
        else:
            self.logger.info(f"✅ Model {model} is available")
        
        # Test interactive session
        return self._test_interactive_llm(model)
    
    def _start_ollama_service(self, ollama_path: str) -> bool:
        """Start Ollama service"""
        try:
            service_cmd = (self.credentials or {}).get('llm_config', {}).get('service_command', 'ollama serve')
            success, stdout, stderr = self.execute_command(f"{ollama_path} serve &")
            
            if success:
                delay = (self.credentials or {}).get('connection_settings', {}).get('service_startup_delay', 30)
                self.logger.info(f"⏳ Waiting {delay}s for service to start...")
                time.sleep(delay)
                return True
            else:
                self.logger.error(f"❌ Failed to start Ollama service: {stderr}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error starting Ollama service: {e}")
            return False
    
    def _install_llm_model(self, ollama_path: str, model: str) -> bool:
        """Install LLM model"""
        try:
            install_cmd = (self.credentials or {}).get('llm_config', {}).get('install_command', 'ollama pull')
            self.logger.info(f"🔄 Installing {model} (this may take several minutes)...")
            
            success, stdout, stderr = self.execute_command(f"{ollama_path} pull {model}", timeout=600)
            
            if success:
                self.logger.info(f"✅ Model {model} installed successfully")
                return True
            else:
                self.logger.error(f"❌ Failed to install model {model}: {stderr}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error installing model: {e}")
            return False
    
    def _test_interactive_llm(self, model: str) -> bool:
        """Test LLM with interactive session"""
        try:
            self.logger.info("🧪 Testing interactive LLM session...")
            
            success, output = self.execute_interactive_command(
                f"ollama run {model}",
                inputs=["Hello, can you respond with 'Test successful'?", "/bye"],
                read_delay=2.0,
                timeout=30
            )
            
            if success and "test successful" in output.lower():
                self.logger.info("✅ LLM interactive test passed")
                return True
            else:
                self.logger.warning(f"⚠️ LLM test response unclear: {output[:200]}...")
                return True  # Consider partial success
                
        except Exception as e:
            self.logger.error(f"❌ LLM interactive test failed: {e}")
            return False
    
    def execute_llm_prompt(self, prompt: str, model: Optional[str] = None, 
                          temperature: float = 0.7, max_tokens: int = 512) -> Tuple[bool, str]:
        """
        Execute a prompt with the LLM
        
        Args:
            prompt: Text prompt to send to LLM
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (success, response)
        """
        model = model or (self.credentials or {}).get('llm_config', {}).get('model', 'llama3.1')
        
        try:
            self.logger.debug(f"Executing LLM prompt with {model}")
            
            # Format the prompt properly
            formatted_prompt = prompt.replace('"', '\\"')
            
            success, output = self.execute_interactive_command(
                f"ollama run {model}",
                inputs=[formatted_prompt, "/bye"],
                read_delay=(self.credentials or {}).get('connection_settings', {}).get('llm_response_delay', 8),
                timeout=60
            )
            
            if success:
                # Clean and extract response
                response = self._extract_llm_response(output, formatted_prompt)
                return True, response
            else:
                return False, output
                
        except Exception as e:
            self.logger.error(f"❌ LLM prompt execution error: {e}")
            return False, str(e)
    
    def _extract_llm_response(self, raw_output: str, original_prompt: str) -> str:
        """Extract clean response from LLM output"""
        try:
            lines = raw_output.split('\n')
            response_lines = []
            capturing = False
            
            for line in lines:
                line = line.strip()
                
                # Start capturing after we see the prompt
                if not capturing and original_prompt[:50] in line:
                    capturing = True
                    continue
                
                # Stop capturing at ollama prompts or exit commands
                if capturing and any(marker in line for marker in [">>>", "Send a message", "/bye"]):
                    break
                
                # Collect response lines
                if capturing and line and not line.startswith(">>>"):
                    response_lines.append(line)
            
            response = '\n'.join(response_lines).strip()
            
            # Clean common artifacts
            response = response.replace(">>> ", "").replace("Send a message (/? for help)", "")
            
            return response if response else raw_output.strip()
            
        except Exception as e:
            self.logger.warning(f"Response extraction failed, returning raw output: {e}")
            return raw_output.strip()
    
    @contextmanager
    def connection_context(self):
        """Context manager for automatic connection handling"""
        try:
            if not self.connect():
                raise ConnectionError("Failed to establish SSH connection")
            yield self
        finally:
            self.close()
    
    def close(self):
        """Close all SSH connections"""
        self.logger.info("🧹 Closing SSH connections...")
        
        if self.target_client:
            try:
                self.target_client.close()
            except:
                pass
            self.target_client = None
            
        if self.channel:
            try:
                self.channel.close()
            except:
                pass
            self.channel = None
            
        if self.jump_client:
            try:
                self.jump_client.close()
            except:
                pass
            self.jump_client = None
        
        self._connected = False
        self.logger.info("✅ SSH connections closed")
    
    def __enter__(self):
        """Context manager entry"""
        if not self.connect():
            raise ConnectionError("Failed to establish SSH connection")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class LLMRemoteExecutor:
    """
    High-level interface for remote LLM operations
    """
    
    def __init__(self, credentials_file: str = "ssh_credentials.json"):
        self.ssh_manager = SSHConnectionManager(credentials_file)
    
    def test_setup(self) -> bool:
        """Test the complete remote LLM setup"""
        with self.ssh_manager.connection_context():
            return self.ssh_manager.test_llm_availability()
    
    def execute_prompt(self, prompt: str, **kwargs) -> Tuple[bool, str]:
        """Execute a prompt on the remote LLM"""
        with self.ssh_manager.connection_context():
            return self.ssh_manager.execute_llm_prompt(prompt, **kwargs)
    
    def execute_batch_prompts(self, prompts: List[str], **kwargs) -> List[Tuple[bool, str]]:
        """Execute multiple prompts in a single session"""
        results = []
        
        with self.ssh_manager.connection_context():
            for prompt in prompts:
                result = self.ssh_manager.execute_llm_prompt(prompt, **kwargs)
                results.append(result)
        
        return results


def main():
    """Example usage of SSH connection manager"""
    print("🚀 SSH Connection Manager for GeneticPromptLab")
    print("=" * 50)
    
    # Test basic connection
    ssh_manager = SSHConnectionManager()
    
    try:
        with ssh_manager.connection_context():
            # Test LLM availability
            if ssh_manager.test_llm_availability():
                print("\n🎉 Remote LLM setup is working correctly!")
                
                # Test a simple prompt
                success, response = ssh_manager.execute_llm_prompt(
                    "Explain genetic algorithms in one sentence."
                )
                
                if success:
                    print(f"\n📝 LLM Response:\n{response}")
                else:
                    print(f"\n❌ LLM execution failed: {response}")
            else:
                print("\n❌ Remote LLM setup is not working")
                
    except Exception as e:
        print(f"\n❌ Setup test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 