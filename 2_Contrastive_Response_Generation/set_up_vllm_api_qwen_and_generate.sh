PORT_NUM=8000
MAX_TRIES=50
try=0
MODEL_NAME_OR_PATH=$1
START=$2
END=$3
SERVICE_UP=false


# Function to check if the web service is running
check_service() {
    curl --silent --fail -X POST "http://0.0.0.0:${PORT_NUM}/v1/chat/completions" \
         -H "Content-Type: application/json" \
         -d '{
                "model": "'"${MODEL_NAME_OR_PATH}"'",
                "messages": [{"role":"user","content":"hello"}],
                "max_tokens": 3076,
                "temperature": 0
             }' > /dev/null
}

while (( try < MAX_TRIES )); do
  echo "Trying port: ${PORT_NUM} (attempt $((try + 1))/${MAX_TRIES})"
  
  # start vllm
  vllm serve ${MODEL_NAME_OR_PATH} --port ${PORT_NUM} \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --reasoning_parser qwen3 \
    --max_model_len 32768 \
    --max-num-seqs 256 &
  WEB_SERVICE_PID=$!
  
  # check if the service is down or not
  sleep 2
  if ! ps -p "${WEB_SERVICE_PID}" >/dev/null 2>&1; then
    echo "Process died immediately (likely port conflict), trying next port..."
    PORT_NUM=$((PORT_NUM + 1))
    try=$((try + 1))
    continue
  fi
  
  # wait until the server is up
  echo "Waiting for the web service to become responsive on port ${PORT_NUM}..."
  for i in $(seq 1 60); do
    if check_service; then
      echo "✓ Web service is up on port ${PORT_NUM}!"
      SERVICE_UP=true
      break
    fi
    
    # process died during startup, break
    if ! ps -p "${WEB_SERVICE_PID}" >/dev/null 2>&1; then
      echo "Process died during startup"
      break
    fi
    
    sleep 10
  done
  
  # service started, break
  if [ "$SERVICE_UP" = true ]; then
    break
  fi
  
  # timeout or service failed, retry
  echo "Service failed to start within timeout, cleaning up and retrying..."
  if ps -p "${WEB_SERVICE_PID}" >/dev/null 2>&1; then
    kill "${WEB_SERVICE_PID}" 2>/dev/null
    wait "${WEB_SERVICE_PID}" 2>/dev/null
  fi
  PORT_NUM=$((PORT_NUM + 1))
  try=$((try + 1))
done

# final check
if [ "$SERVICE_UP" != true ]; then
  echo "ERROR: Failed to start web service after ${MAX_TRIES} attempts"
  exit 1
fi

echo "Web service is up! Running Python script..."

# use the port number that successfully starts
CUDA_VISIBLE_DEVICES=0 bash exec_synthesize_assistant_response_using_qwen.sh \
  ${MODEL_NAME_OR_PATH} $START $END "http://0.0.0.0:${PORT_NUM}/v1"

# clean up the vllm service
echo "Cleaning up vLLM service..."
kill "${WEB_SERVICE_PID}" 2>/dev/null
wait "${WEB_SERVICE_PID}" 2>/dev/null
echo "Done."
