import os
import signal
import subprocess
import sys
import time
import threading


class TrainingManager:
    def __init__(self):
        self.current_process = None
        self.should_stop = False
        self.server_started = False
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self.handle_sigterm)
        signal.signal(signal.SIGINT, self.handle_sigterm)

    def handle_sigterm(self, signum, frame):
        """Handle termination signals gracefully"""
        print("Received termination signal. Starting graceful shutdown...")
        self.should_stop = True
        if self.current_process:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=115)  # Wait just under 2 minutes
            except subprocess.TimeoutExpired:
                self.current_process.kill()

    def monitor_output(self, pipe):
        """Monitor process output for server start message"""
        for line in iter(pipe.readline, b''):
            line_str = line.decode('utf-8').strip()
            print(line_str)  # Echo the output
            if "Started server process" in line_str:
                self.server_started = True
                break

    def run_server(self, command, env=None):
        """Run the vllm server and wait for it to start"""
        if self.should_stop:
            return False

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        try:
            self.current_process = subprocess.Popen(
                command,
                env=merged_env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=False
            )

            # Start monitoring the output in a separate thread
            monitor_thread = threading.Thread(
                target=self.monitor_output, 
                args=(self.current_process.stdout,)
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            # Wait for server to start or timeout
            timeout = 300  # 5 minutes timeout
            start_time = time.time()
            while not self.server_started and not self.should_stop:
                if time.time() - start_time > timeout:
                    print("Timeout waiting for server to start")
                    self.current_process.terminate()
                    return False
                time.sleep(1)

            return True

        except Exception as e:
            print(f"Error running server: {e}")
            return False

    def run_command(self, command, env=None):
        """Run a command and handle its execution"""
        if self.should_stop:
            return False

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        try:
            self.current_process = subprocess.Popen(
                command,
                env=merged_env,
                shell=True
            )
            self.current_process.wait()
            
            if self.current_process.returncode != 0:
                print(f"Command failed with return code {self.current_process.returncode}")
                return False
            return True
        except Exception as e:
            print(f"Error running command: {e}")
            return False

    def run_training(self):
        """Main training execution flow"""
        # Read environment variables or config files
        vllm_devices = os.getenv('VLLM_DEVICES', '0')
        training_devices = os.getenv('TRAINING_DEVICES', '0')
        tensor_parallel_size = len(vllm_devices.split(','))

        # Step 1: Run vllm-serve
        vllm_env = {
            'CUDA_VISIBLE_DEVICES': vllm_devices
        }
        vllm_command = (
            f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 trl vllm-serve --model /opt/ml/input/data/model "
            f"--tensor-parallel-size {tensor_parallel_size} "
            f"--gpu_memory_utilization 0.9 "
        )
        
        # Start the server and wait for it to be ready
        server_process = None
        if not self.run_server(vllm_command, vllm_env):
            sys.exit(1)

        # Step 2: Run training
        if self.should_stop:
            return

        training_env = {
            'CUDA_VISIBLE_DEVICES': training_devices
        }
        training_command = (
            "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch "
            "--config_file /opt/ml/input/data/config/accelerate_config.yaml "
            "/home/open-r1/open-r1-sagemaker/src/open_r1/grpo_customized.py "
            "--config /opt/ml/input/data/config/training_config.yaml"
        )
        
        if not self.run_command(training_command, training_env):
            sys.exit(1)

        # After training is done, terminate the server
        if self.current_process:
            self.current_process.terminate()

def main():
    manager = TrainingManager()
    manager.run_training()

if __name__ == "__main__":
    main()