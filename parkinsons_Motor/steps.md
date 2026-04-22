Next steps:
  cd /Users/vinay/vscode/hackathon/meta-hackathon/environment/parkinsons_Motor
  # Edit your environment implementation in server/parkinsons_Motor_environment.py
  # Edit your models in models.py
  # Install dependencies: uv sync

  # To integrate into OpenEnv repo:
  # 1. Copy this directory to <repo_root>/envs/parkinsons_Motor_env
  # 2. Build from repo root: docker build -t parkinsons_Motor_env:latest -f 
envs/parkinsons_Motor_env/server/Dockerfile .
  # 3. Run your image: docker run -p 8000:8000 parkinsons_Motor_env:latest

to run the server - uv run --project . server